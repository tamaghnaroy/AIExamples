from typing import Dict, Any, List
import sys
import os
import logging
import threading
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent
from ProductDesigner.utils.timeout_decorator import timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SafeProductInterviewerAgent(BaseAgent):
    """
    A safer, modular version of the ProductInterviewerAgent.
    It uses a centralized LLMService and focuses on the interview logic.
    """

    def __init__(self):
        """Initialize the Safe Product Interviewer Agent."""
        super().__init__()
        self.web_search_enabled = os.getenv('DISABLE_WEB_SEARCH', 'false').lower() != 'true'
        self.search_tool = DuckDuckGoSearchRun() if self.web_search_enabled else None
        if self.web_search_enabled:
            logging.info("Web search is enabled.")
        else:
            logging.info("Web search is disabled.")

        self.questions = self._get_planning_questions()
        self.current_question_idx = 0
        self.recommendations_cache = {}
        self.processing_status = {}
        self.output_parser = None

    def _get_planning_questions(self) -> List[str]:
        """Return the 13 planning questions for product discovery."""
        return [
            "What is the primary purpose of this application or system?",
            "Who are the main users or target audience of this application?",
            "What are the 3-5 core features needed for the MVP (Minimum Viable Product)?",
            "What specific problem does this solution solve for your users?",
            "Are there any existing solutions to this problem? How is yours different?",
            "What technologies or tech stack do you have in mind? Or would you like recommendations?",
            "What are the main user flows or journeys through the application?",
            "Are there any specific UI/UX requirements or preferences?",
            "What are your plans for data storage and management?",
            "Are there any specific security requirements or concerns?",
            "What is your timeline for development of the MVP?",
            "Are there any third-party integrations needed?",
            "What metrics would define success for this product?"
        ]

    def get_next_question(self) -> str:
        """Get the next question to ask the user."""
        if self.current_question_idx < len(self.questions):
            return self.questions[self.current_question_idx]
        return "All questions answered. Proceeding to the next step."

    @timeout(20)
    def search_web_for_context(self, query: str) -> str:
        """Use web search to provide context for questions with timeout protection."""
        if not self.search_tool:
            logging.info("Web search is disabled. Skipping.")
            return "Web search is disabled."
        
        logging.info(f"Performing web search for: {query}")
        try:
            results = self.search_tool.run(query)
            logging.info("Web search successful.")
            return results
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            return f"Search failed with error: {e}"

    def _run_llm_step(self, state: Dict[str, Any], prompt_template_str: str, params: Dict[str, Any], fallback_response: str, 
    output_parser: str = None, model_override:str=None, temparature_override:float=None) -> str:
        """Helper to run a single LLM step using the base agent's service."""
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        full_params = {**state, **params}
        return self.llm_service.execute(prompt_template, full_params, fallback_response, output_parser, model_override, temparature_override)

    @timeout(30)
    def generate_initial_answer(self, state: Dict[str, Any], question: str, search_results: str = "") -> str:
        logging.info(f"Generating initial answer for: {question}")
        template = (
            "You are an expert product manager. Based on the initial idea and any relevant search results, "
            "provide a comprehensive and insightful answer to the following question.\n\n"
            "Initial Idea: {initial_idea}\n"
            "Question: {question}\n"
            "Search Results: {search_results}\n\n"
            "Include reasoning, multiple options where applicable, and industry best practices.\n"
            "Answer:"
        )
        params = {"question": question, "search_results": search_results}
        return self._run_llm_step(state, template, params, "Could not generate an initial answer due to an error.")

    @timeout(30)
    def critique_answer(self, state: Dict[str, Any], question: str, generated_answer: str) -> str:
        logging.info(f"Critiquing answer for: {question}")
        template = (
            "You are a meticulous and critical product director. Review the following answer to the question, "
            "considering the initial product idea. Identify weaknesses, missing information, or potential issues.\n\n"
            "Initial Idea: {initial_idea}\n"
            "Question: {question}\n"
            "Generated Answer: {generated_answer}\n\n"
            "Provide a concise, constructive critique.\n"
            "Critique:"
        )
        params = {"question": question, "generated_answer": generated_answer}
        return self._run_llm_step(state, template, params, "Could not critique the answer due to an error.")

    @timeout(30)
    def refine_answer(self, state: Dict[str, Any], question: str, original_answer: str, critique: str) -> str:
        logging.info(f"Refining answer for: {question}")
        template = (
            "You are an expert product manager revising your work based on feedback. "
            "Below is a question, an original answer, and a critique.\n\n"
            "Question: {question}\n"
            "Original Answer: {original_answer}\n"
            "Critique: {critique}\n\n"
            "Your task is to integrate the feedback to create an improved answer. Address all points in the critique.\n"
            "New, Refined Answer:"
        )
        params = {"question": question, "original_answer": original_answer, "critique": critique}
        return self._run_llm_step(state, template, params, "Could not refine the answer due to an error.")

    @timeout(30)
    def present_to_user(self, state: Dict[str, Any], question: str, refined_answer: str) -> str:
        logging.info(f"Formatting answer for user presentation: {question}")
        template = (
            "You are a helpful AI assistant. Reformat the following answer into a clear, user-friendly presentation. "
            "Use markdown for readability (e.g., bullet points, bold text). "
            "The user needs to understand this to confirm if the project is on the right track.\n\n"
            "Question: {question}\n"
            "Answer: {refined_answer}\n\n"
            "Start with a friendly opening.\n"
            "Presentation:"
        )
        params = {"question": question, "refined_answer": refined_answer}
        return self._run_llm_step(state, template, params, "Could not format the answer for presentation due to an error.")

    def process_question_with_self_critique(self, state: Dict[str, Any], question: str, progress_callback=None) -> Dict[str, str]:
        """Process a question through the full self-critique pipeline with optional progress updates."""
        logging.info(f"Processing question with self-critique: {question}")
        initial_idea = state.get('initial_idea', '')

        def update_progress(message, status, prog):
            if progress_callback: progress_callback(message, status, prog)

        try:
            update_progress("Starting analysis...", "processing", 0)
            
            search_query = f"User problem and target audience for a product based on the idea: {initial_idea}"
            search_results = self.search_web_for_context(search_query)
            update_progress("Web search complete.", "processing", 20)

            initial_answer = self.generate_initial_answer(state, question, search_results)
            update_progress("Initial answer generated.", "processing", 40)

            critique = self.critique_answer(state, question, initial_answer)
            update_progress("Answer critiqued.", "processing", 60)

            refined_answer = self.refine_answer(state, question, initial_answer, critique)
            update_progress("Answer refined.", "processing", 80)

            user_presentation = self.present_to_user(state, question, refined_answer)
            update_progress("Formatted for presentation.", "completed", 100)

            return {
                "refined_answer": refined_answer,
                "user_presentation": user_presentation,
                "search_results": search_results,
                "critique": critique,
            }
        except Exception as e:
            logging.error(f"Error in self-critique process for question '{question}': {e}")
            update_progress(f"An error occurred: {e}", "error", 100)
            return {"error": str(e)}

    def get_question_recommendation(self, question_number: int, state: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """Get recommendation for a specific question number, with optional progress updates."""
        initial_idea = state.get('initial_idea', '')
        cache_key = f"{question_number}_{hash(initial_idea)}"
        
        if cache_key in self.recommendations_cache:
            logging.info(f"Returning cached recommendation for question {question_number}")
            if progress_callback: progress_callback("Recommendation loaded from cache.", "completed", 100)
            return self.recommendations_cache[cache_key]

        try:
            question = self.questions[question_number - 1]
            self.processing_status[question_number] = 'processing'

            recommendation = self.process_question_with_self_critique(state, question, progress_callback)

            if "error" not in recommendation:
                self.recommendations_cache[cache_key] = recommendation
                self.processing_status[question_number] = 'completed'
            else:
                self.processing_status[question_number] = 'error'

            return recommendation

        except Exception as e:
            logging.error(f"Error processing question '{question_number}': {e}")
            self.processing_status[question_number] = 'error'
            if progress_callback: progress_callback(f"An error occurred: {e}", "error", 100)
            return {"error": str(e)}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state by answering the next question."""
        try:
            qna_history = state.get('qna_history', {}).copy()
            next_question_text = self.get_next_question()

            if "All questions answered" in next_question_text:
                return {"qna_history": qna_history}

            current_question = self.questions[self.current_question_idx]
            
            question_analysis = self.process_question_with_self_critique(state, current_question)
            
            if "error" in question_analysis:
                logging.error(f"Error processing question '{current_question}': {question_analysis['error']}")
                return {"qna_history": qna_history}

            qna_history[current_question] = {
                "llm_recommendation": question_analysis.get("refined_answer"),
                "user_presentation": question_analysis.get("user_presentation"),
                "search_context": question_analysis.get("search_results"),
                "internal_critique": question_analysis.get("critique"),
                "user_response": "[User would verify/modify the recommendation here]"
            }
            
            self.current_question_idx += 1
            return {"qna_history": qna_history}
            
        except Exception as e:
            logging.error(f"Error in SafeProductInterviewerAgent.run: {e}")
            return {"qna_history": state.get('qna_history', {})}

    def is_recommendation_ready(self, question_number: int, initial_idea: str) -> bool:
        """Check if recommendation for a question is ready."""
        return f"{question_number}_{hash(initial_idea)}" in self.recommendations_cache
    
    def get_processing_status(self, question_number: int) -> str:
        """Get the processing status of a question ('processing', 'completed', 'error', or None)."""
        return self.processing_status.get(question_number)

    def prefetch_next_recommendation(self, current_question_number: int, state: Dict[str, Any]):
        """
        Prefetches the recommendation for the next question in a background thread.
        """
        next_question_number = current_question_number + 1
        if next_question_number > len(self.questions):
            logging.info("No more questions to prefetch.")
            return

        initial_idea = state.get('initial_idea', '')
        
        # Check if already cached or processing
        if self.is_recommendation_ready(next_question_number, initial_idea):
            logging.info(f"Prefetch: Recommendation for question {next_question_number} is already cached.")
            return
            
        if self.get_processing_status(next_question_number) == 'processing':
            logging.info(f"Prefetch: Recommendation for question {next_question_number} is already being processed.")
            return

        logging.info(f"Starting prefetch for question {next_question_number}")

        def prefetch_task():
            logging.info(f"Background prefetch task started for question {next_question_number}.")
            try:
                # We don't need a progress callback for prefetching as it's a background task
                self.get_question_recommendation(next_question_number, state, progress_callback=None)
                logging.info(f"Background prefetch task for question {next_question_number} completed.")
            except Exception as e:
                logging.error(f"Error during prefetch for question {next_question_number}: {e}")

        thread = threading.Thread(target=prefetch_task)
        thread.daemon = True  # Allows main program to exit even if threads are running
        thread.start()
