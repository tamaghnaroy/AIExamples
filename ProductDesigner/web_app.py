import os
import sys
import logging
import traceback
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .deep_planning_graph import create_deep_planning_graph, initialize_state
from .graph_state import GraphState
from .agents.safe_product_interviewer import SafeProductInterviewerAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active sessions and their states
active_sessions = {}

class WebAppInterface:
    """Web interface for the Deep Planning LangGraph system."""
    
    def __init__(self):
        self.graph = create_deep_planning_graph()
    
    def create_session(self, session_id: str, initial_idea: str) -> Dict[str, Any]:
        """Create a new planning session."""
        state = initialize_state(initial_idea)
        active_sessions[session_id] = {
            'state': state,
            'current_step': 'product_interviewer',
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'interviewer': SafeProductInterviewerAgent(),
            'current_question_idx': 0
        }
        return active_sessions[session_id]
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get an existing session."""
        return active_sessions.get(session_id)
    
    def update_session_state(self, session_id: str, updates: Dict[str, Any]):
        """Update session state."""
        if session_id in active_sessions:
            active_sessions[session_id]['state'].update(updates)

web_interface = WebAppInterface()

@app.route('/')
def index():
    """Main page of the web application."""
    return render_template('index.html')

@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    """Dashboard for monitoring the planning process."""
    session_data = web_interface.get_session(session_id)
    if not session_data:
        return "Session not found", 404
    return render_template('dashboard.html', session_id=session_id)

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start a new planning session."""
    data = request.get_json()
    initial_idea = data.get('initial_idea', '')
    
    if not initial_idea.strip():
        return jsonify({'error': 'Initial idea is required'}), 400
    
    session_id = str(uuid.uuid4())
    session_data = web_interface.create_session(session_id, initial_idea)
    
    return jsonify({
        'session_id': session_id,
        'status': 'created',
        'initial_idea': initial_idea,
        'created_at': session_data['created_at']
    })

@app.route('/api/session/<session_id>/status')
def get_session_status(session_id):
    """Get the current status of a planning session."""
    session_data = web_interface.get_session(session_id)
    if not session_data:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'session_id': session_id,
        'current_step': session_data['current_step'],
        'status': session_data['status'],
        'state': session_data['state'],
        'created_at': session_data['created_at']
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Deep Planning System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Join a planning session for real-time updates."""
    session_id = data.get('session_id')
    if session_id and session_id in active_sessions:
        emit('session_joined', {'session_id': session_id})
        # Join the session room to receive targeted updates
        join_room(session_id)
        logger.info(f"Client {request.sid} joined session {session_id}")
    else:
        emit('error', {'message': 'Invalid session ID'})
        logger.warning(f"Client {request.sid} failed to join session {session_id} - invalid ID")

@socketio.on('start_interview')
def handle_start_interview(data):
    """Start the interview process."""
    session_id = data.get('session_id')
    session_data = web_interface.get_session(session_id)
    
    if not session_data:
        logger.error(f"Could not start interview for session {session_id}: Session not found.")
        emit('error', {'message': 'Session not found'})
        return
    
    logger.info(f"Starting interview for session {session_id}")
    emit('interview_started', {'message': 'Interview process has started.'})
    
    # Store the client session ID for this session
    client_sid = request.sid
    session_data['client_sid'] = client_sid
    session_data['current_question_idx'] = 0
    session_data['awaiting_user_response'] = False
    
    emit('step_started', {
        'step': 'product_interviewer',
        'message': 'Starting product interview process...'
    })
    
    # Process the first question
    process_next_question(session_id)

def process_next_question(session_id):
    """Process the next question in the interview using just-in-time fetching."""
    session_data = web_interface.get_session(session_id)
    if not session_data:
        return
    
    interviewer = session_data['interviewer']
    state = session_data['state']
    current_idx = session_data['current_question_idx']
    client_sid = session_data.get('client_sid')
    
    # Check if we've completed all questions
    if current_idx >= len(interviewer.questions):
        logger.info(f"Interview complete for session {session_id}.")
        session_data['current_step'] = 'interview_completed'
        socketio.emit('interview_completed', {
            'step': 'product_interviewer',
            'message': 'Product interview completed successfully!',
            'next_step': 'user_review'
        }, room=client_sid)
        return
    
    question_number = current_idx + 1
    question = interviewer.questions[current_idx]
    initial_idea = state['initial_idea']
    
    logger.info(f"Processing question {question_number}: {question[:50]}...")
    
    # Mark as awaiting user response
    session_data['awaiting_user_response'] = True
    
    # Emit that we're starting to process this question
    socketio.emit('question_processing', {
        'question_number': question_number,
        'total_questions': len(interviewer.questions),
        'question': question,
        'status': 'processing'
    }, room=client_sid)
    
    def fetch_and_emit_recommendation():
        try:
            logger.info(f"Fetching recommendation for question {question_number}")

            # The SafeProductInterviewerAgent expects a GraphState object.
            current_graph_state = GraphState(**state)

            recommendation_data = interviewer.get_question_recommendation(
                question_number=question_number,
                state=current_graph_state,
                progress_callback=lambda step, msg, prog: socketio.emit('llm_progress', {
                    'question_number': question_number,
                    'step': step,
                    'message': msg,
                    'progress': prog
                }, room=client_sid)
            )

            if "error" in recommendation_data:
                raise Exception(recommendation_data["error"])

            # Update session state with all parts of the analysis
            state['qna_history'][question] = {
                "llm_recommendation": recommendation_data.get("refined_answer", ""),
                "user_presentation": recommendation_data.get("user_presentation", ""),
                "search_context": recommendation_data.get("search_results", ""),
                "internal_critique": recommendation_data.get("critique", ""),
                "user_response": "[Pending user verification]"
            }
            
            # Emit the completed question with the user-facing presentation
            socketio.emit('question_completed', {
                'question_number': question_number,
                'question': question,
                'analysis': recommendation_data, # Send the whole analysis
                'status': 'completed'
            }, room=client_sid)

            # Start prefetching next question in the background
            if question_number < len(interviewer.questions):
                logger.info(f"Starting prefetch for question {question_number + 1}")
                # Pass the current graph state to the prefetch method
                interviewer.prefetch_next_recommendation(question_number, current_graph_state)

        except Exception as e:
            logger.error(f"Error processing question {question_number}: {str(e)}")
            logger.error(traceback.format_exc())
            session_data['awaiting_user_response'] = False
            # Ensure this emit is also thread-safe
            socketio.emit('question_error', {
                'question_number': question_number,
                'question': question,
                'error': str(e)
            }, room=client_sid)
    
    # Run in background thread
    thread = threading.Thread(target=fetch_and_emit_recommendation)
    thread.daemon = True
    thread.start()

@socketio.on('user_response')
def handle_user_response(data):
    """Handle user response to a question."""
    session_id = data.get('session_id')
    question_number = data.get('question_number')
    user_response = data.get('response')
    
    session_data = web_interface.get_session(session_id)
    if not session_data:
        logger.error(f"Session {session_id} not found during question processing.")
        emit('error', {'message': 'Session not found'})
        return
    
    # Check if we're currently awaiting a user response
    if not session_data.get('awaiting_user_response', False):
        logger.warning(f"Received user response for session {session_id} when not awaiting one.")
        emit('error', {'message': 'Not currently awaiting user response'})
        return
    
    # Update the user response in the session state
    questions = list(session_data['state']['qna_history'].keys())
    if question_number <= len(questions):
        question = questions[question_number - 1]
        session_data['state']['qna_history'][question]['user_response'] = user_response
        
        logger.info(f"User response for q{question_number} in session {session_id} saved.")
        
        emit('response_saved', {
            'question_number': question_number,
            'response': user_response,
            'message': 'Response saved successfully'
        })
        
        # Mark that we're no longer awaiting user response
        session_data['awaiting_user_response'] = False
        
        # Move to next question
        session_data['current_question_idx'] += 1
        
        # Process next question after a short delay
        import time
        def continue_interview():
            time.sleep(0.5)  # Brief pause before next question
            process_next_question(session_id)
        
        thread = threading.Thread(target=continue_interview)
        thread.daemon = True
        thread.start()

@socketio.on('continue_to_next_step')
def handle_continue_to_next_step(data):
    """Continue to the next step in the planning process."""
    session_id = data.get('session_id')
    session_data = web_interface.get_session(session_id)
    
    if not session_data:
        logger.error(f"Could not continue to next step for session {session_id}: Session not found.")
        emit('error', {'message': 'Session not found'})
        return
    
    current_step = session_data['current_step']
    
    if current_step == 'interview_completed':
        # Move to PRD generation
        session_data['current_step'] = 'prd_generation'
        logger.info(f"Starting PRD generation for session {session_id}.")
        emit('step_started', {
            'step': 'prd_generation',
            'message': 'Generating Product Requirements Document...'
        })
        
        def run_generation_step(step_name, completion_step, next_step_name, output_key):
            try:
                logger.info(f"Running generation step '{step_name}' for session {session_id}.")
                # Run the graph for one step
                updated_state = web_interface.graph.invoke(session_data['state'], config={'run_name': step_name})
                session_data['state'] = updated_state
                session_data['current_step'] = completion_step

                logger.info(f"Generation step '{step_name}' for session {session_id} completed.")
                emit('step_completed', {
                    'step': step_name,
                    'message': f'{step_name.replace("_", " ").title()} generated successfully!',
                    'next_step': next_step_name,
                    'document': updated_state.get(output_key, '')
                })

                # Automatically start the next step if there is one
                if next_step_name:
                    socketio.emit('start_next_step', {'next_step': next_step_name})

            except Exception as e:
                logger.error(f"Error during {step_name} for session {session_id}: {e}")
                logger.error(traceback.format_exc())
                emit('error', {'message': f'An error occurred during {step_name}.'})

        # This will be a sequential process now
        if current_step == 'interview_completed':
            thread = threading.Thread(target=run_generation_step, args=('prd_generation', 'prd_completed', 'tech_design', 'prd_document'))
            thread.daemon = True
            thread.start()
        elif current_step == 'prd_completed':
            thread = threading.Thread(target=run_generation_step, args=('tech_design', 'tech_design_completed', None, 'tech_design_document'))
            thread.daemon = True
            thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
