from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import with fallback for different execution contexts
try:
    from ..graph_state import GraphState
    from .base_agent import BaseAgent
except (ImportError, ValueError):
    # Fallback for when running tests or as standalone module
    from graph_state import GraphState
    from agents.base_agent import BaseAgent

class ProductInterviewerAgent(BaseAgent):
    """
    Acts as a combined Product Manager and Technical Lead to guide the user
    through a structured discovery process for their product idea.
    Enhanced with self-answering, web search, and self-critiquing capabilities.
    """
    
    def __init__(self, model: str = "gpt-4-turbo", temperature: float = 0.1):
        """Initialize the Product Interviewer Agent."""
        super().__init__(model, temperature)
        self.search_tool = DuckDuckGoSearchRun()
        self.questions = self._get_planning_questions()
        self.current_question_idx = 0
        
        # Template for generating initial answers with web search
        self.answer_generation_template = PromptTemplate.from_template(
            """You are an expert Product Manager analyzing a project idea to provide comprehensive answers.
            
            Project Idea: {initial_idea}
            
            Question to Answer: {question}
            
            Web Search Results (if available): {search_results}
            
            Your task is to provide a detailed, well-reasoned answer to this question based on:
            1. The project idea context
            2. Industry best practices
            3. Web search results (if provided)
            4. Your expertise in product management
            
            Provide a comprehensive answer that includes:
            - Direct answer to the question
            - Reasoning behind your recommendations
            - Multiple options where applicable
            - Industry standards and best practices
            
            Answer:
            """
        )
        
        # Template for self-critiquing the generated answer
        self.critique_template = PromptTemplate.from_template(
            """You are a senior product consultant reviewing an answer provided by a product manager.
            
            Project Idea: {initial_idea}
            Question: {question}
            Generated Answer: {generated_answer}
            
            Critically evaluate this answer on the following criteria:
            1. Completeness - Does it fully address the question?
            2. Accuracy - Is the information correct and up-to-date?
            3. Relevance - Is it specifically relevant to this project idea?
            4. Actionability - Does it provide clear, actionable guidance?
            5. Options - Does it present multiple viable options where appropriate?
            
            Provide your critique and suggest improvements:
            
            Critique:
            """
        )
        
        # Template for refining the answer based on critique
        self.refinement_template = PromptTemplate.from_template(
            """You are refining a product management answer based on expert critique.
            
            Project Idea: {initial_idea}
            Question: {question}
            Original Answer: {original_answer}
            Critique: {critique}
            
            Based on the critique, provide an improved, refined answer that addresses the identified issues.
            Make sure the refined answer is:
            - Complete and comprehensive
            - Accurate and current
            - Highly relevant to the project
            - Actionable and specific
            - Includes multiple options where beneficial
            
            Refined Answer:
            """
        )
        
        # Template for presenting the final recommendation to the user
        self.user_presentation_template = PromptTemplate.from_template(
            """You are presenting a well-researched recommendation to a user for their project.
            
            Project Idea: {initial_idea}
            Question: {question}
            Refined Answer: {refined_answer}
            
            Present this information to the user in a friendly, professional manner that:
            1. Clearly states the question being addressed
            2. Provides your recommended answer with reasoning
            3. Offers alternative options where applicable
            4. Asks for their feedback and verification
            5. Invites them to modify or accept the recommendation
            
            Format your response as a conversation with the user, making it easy for them to
            understand and respond to your recommendations.
            
            User Presentation:
            """
        )
    
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
    
    def _format_qa_history(self, qna_history: Dict[str, Any]) -> str:
        """Format the Q&A history for display in the prompt."""
        if not qna_history:
            return "No questions have been asked yet."
        
        formatted = ""
        for idx, (question, answer) in enumerate(qna_history.items()):
            formatted += f"Q{idx+1}: {question}\n"
            formatted += f"A{idx+1}: {answer}\n\n"
        
        return formatted
    
    def get_next_question(self) -> str:
        """Get the next question to ask the user."""
        if self.current_question_idx >= len(self.questions):
            return "All questions have been answered. Ready to generate the PRD."
        
        return f"Question {self.current_question_idx + 1}: {self.questions[self.current_question_idx]}"
    
    def search_web_for_context(self, query: str) -> str:
        """Use web search to provide context for questions."""
        try:
            results = self.search_tool.run(query)
            return results
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def generate_initial_answer(self, question: str, initial_idea: str, search_results: str = "") -> str:
        """Generate an initial answer to the question using LLM and web search results."""
        chain = (
            {"initial_idea": RunnablePassthrough(), "question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
            | self.answer_generation_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "initial_idea": initial_idea,
            "question": question,
            "search_results": search_results
        })
    
    def critique_answer(self, question: str, initial_idea: str, generated_answer: str) -> str:
        """Perform self-critique on the generated answer."""
        chain = (
            {"initial_idea": RunnablePassthrough(), "question": RunnablePassthrough(), "generated_answer": RunnablePassthrough()}
            | self.critique_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "initial_idea": initial_idea,
            "question": question,
            "generated_answer": generated_answer
        })
    
    def refine_answer(self, question: str, initial_idea: str, original_answer: str, critique: str) -> str:
        """Refine the answer based on the critique."""
        chain = (
            {"initial_idea": RunnablePassthrough(), "question": RunnablePassthrough(), 
             "original_answer": RunnablePassthrough(), "critique": RunnablePassthrough()}
            | self.refinement_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "initial_idea": initial_idea,
            "question": question,
            "original_answer": original_answer,
            "critique": critique
        })
    
    def present_to_user(self, question: str, initial_idea: str, refined_answer: str) -> str:
        """Present the refined answer to the user for verification."""
        chain = (
            {"initial_idea": RunnablePassthrough(), "question": RunnablePassthrough(), "refined_answer": RunnablePassthrough()}
            | self.user_presentation_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "initial_idea": initial_idea,
            "question": question,
            "refined_answer": refined_answer
        })
    
    def process_question_with_self_critique(self, question: str, initial_idea: str) -> Dict[str, str]:
        """Process a question through the full self-critique pipeline."""
        # Step 1: Perform web search for context
        search_query = f"{question} best practices {initial_idea}"
        search_results = self.search_web_for_context(search_query)
        
        # Step 2: Generate initial answer
        initial_answer = self.generate_initial_answer(question, initial_idea, search_results)
        
        # Step 3: Self-critique the answer
        critique = self.critique_answer(question, initial_idea, initial_answer)
        
        # Step 4: Refine the answer based on critique
        refined_answer = self.refine_answer(question, initial_idea, initial_answer, critique)
        
        # Step 5: Present to user
        user_presentation = self.present_to_user(question, initial_idea, refined_answer)
        
        return {
            "question": question,
            "search_results": search_results,
            "initial_answer": initial_answer,
            "critique": critique,
            "refined_answer": refined_answer,
            "user_presentation": user_presentation
        }
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Process the current state and guide the user through the enhanced interview process."""
        # Initialize qna_history if it doesn't exist
        qna_history = state.get('qna_history', {})
        initial_idea = state.get('initial_idea', '')
        
        # Get the current question
        next_question = self.get_next_question()
        
        # If we're done with questions, return the completed state
        if next_question.startswith("All questions"):
            return {"qna_history": qna_history}
        
        # Get the current question text
        current_question = self.questions[self.current_question_idx]
        
        # Process the question through the self-critique pipeline
        question_analysis = self.process_question_with_self_critique(current_question, initial_idea)
        
        # In a real implementation, this would present the user_presentation to the user
        # and wait for their feedback. For this demo, we'll store the refined answer.
        
        # Store the processed question and refined answer in the history
        qna_history[current_question] = {
            "llm_recommendation": question_analysis["refined_answer"],
            "user_presentation": question_analysis["user_presentation"],
            "search_context": question_analysis["search_results"],
            "internal_critique": question_analysis["critique"],
            "user_response": "[User would verify/modify the recommendation here]"
        }
        
        # Move to the next question
        self.current_question_idx += 1
        
        # Return the updated state
        return {"qna_history": qna_history}
