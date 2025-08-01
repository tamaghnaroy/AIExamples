import os
import json
from dotenv import load_dotenv
from deep_planning_graph import create_deep_planning_graph, initialize_state
from agents.product_interviewer import ProductInterviewerAgent
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class HumanInTheLoopInterface:
    """Mock human-in-the-loop interface for interacting with the user."""
    
    def get_initial_idea(self) -> str:
        """Get the initial project idea from the user."""
        print("\n=== Welcome to the Deep Planning System ===")
        print("This system will help you transform your idea into a developer-ready blueprint.")
        print("\nPlease describe your project idea:")
        
        # In a real application, this would get input from the user
        # For demonstration, we'll use a mock idea
        mock_idea = "A web application that helps users track their daily habits and visualize progress over time"
        print(f"\nDEMO MODE: Using mock idea: '{mock_idea}'")
        return mock_idea
    
    def ask_question(self, question: str) -> str:
        """Ask the user a question and get their response."""
        print(f"\n>>> {question}")
        
        # In a real application, this would get input from the user
        # For demonstration, we'll use mock answers
        mock_answers = {
            "What is the primary purpose of this application or system?": 
                "To help users build positive habits by tracking daily activities and visualizing their progress",
            "Who are the main users or target audience of this application?":
                "Individuals looking to build better habits, typically aged 25-45",
            "What are the 3-5 core features needed for the MVP?":
                "1. Habit tracking with daily check-ins, 2. Progress visualization with charts, 3. Reminder system, 4. Simple user authentication",
            "What specific problem does this solution solve for your users?":
                "It helps users stay consistent with their habits by providing visual feedback and accountability",
            "Are there any existing solutions to this problem? How is yours different?":
                "Yes, there are apps like Habitica and Streaks. Ours will focus on simplicity and beautiful data visualization",
            "What technologies or tech stack do you have in mind?":
                "I'd like recommendations for a modern web stack",
            "What are the main user flows or journeys through the application?":
                "Sign up, create habits, daily check-ins, view progress, set reminders",
            "Are there any specific UI/UX requirements or preferences?":
                "Clean, minimalist design with a focus on visual data representation",
            "What are your plans for data storage and management?":
                "Not sure, need recommendations",
            "Are there any specific security requirements or concerns?":
                "Basic user authentication and data privacy",
            "What is your timeline for development of the MVP?":
                "3 months",
            "Are there any third-party integrations needed?":
                "Calendar integration for reminders would be nice",
            "What metrics would define success for this product?":
                "User retention rate, habit completion rate, user satisfaction"
        }
        
        # Get the question number (if any) to match with our mock answers
        for q in mock_answers:
            if q in question:
                answer = mock_answers[q]
                print(f"DEMO MODE: Using mock answer: '{answer}'")
                return answer
                
        return "I'm not sure, what would you recommend?"
    
    def review_documents(self, prd: str, tdd: str) -> tuple[bool, str]:
        """Let the user review the PRD and TDD documents."""
        print("\n=== Document Review ===")
        print("PRD and TDD documents have been generated.")
        print("In a real application, you would review them here.")
        
        # In a real application, the user would review and provide feedback
        # For demonstration, we'll auto-approve
        print("DEMO MODE: Auto-approving documents")
        return True, ""
    
    def display_final_package(self, final_package: Dict[str, str]) -> None:
        """Display the final project blueprint package to the user."""
        print("\n=== Final Project Blueprint ===")
        print("Your project blueprint has been generated!")
        
        for filename, content in final_package.items():
            print(f"\n--- {filename} ---")
            # In a real app, we'd save these files or display them properly
            print(f"Content length: {len(content)} characters")
            print(f"First 100 chars: {content[:100]}...")
            
            # Save the files to disk
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved to {filename}")
        
        print("\nAll files have been saved to disk. Your project blueprint is ready!")

def main():
    """Main entry point for the Deep Planning LangGraph application."""
    # Create the user interface
    ui = HumanInTheLoopInterface()
    
    # Get the initial idea from the user
    initial_idea = ui.get_initial_idea()
    
    # Initialize the graph state
    state = initialize_state(initial_idea)
    
    # Create the graph
    graph = create_deep_planning_graph()
    
    print("\n=== Starting Deep Planning Process ===")
    
    # In a real implementation, this would be properly integrated with the LangGraph
    # For this demonstration, we'll simulate the flow
    
    # Simulate Product Interviewer Agent
    print("\n--- Product Interviewer Stage ---")
    interviewer = ProductInterviewerAgent()
    
    # Ask the 13 planning questions
    for i in range(len(interviewer.questions)):
        question = interviewer.questions[i]
        answer = ui.ask_question(question)
        state["qna_history"][question] = answer
    
    # Run the graph with appropriate event handlers for user interaction
    # In a real implementation, we would integrate the UI with the graph execution
    
    # For this demonstration, we'll pretend we ran the graph and got results
    print("\n=== Deep Planning Process Complete ===")
    print("In a real implementation, this would run the full LangGraph.")
    print("For demonstration, we're simulating the final output.")
    
    # Simulate final output
    mock_final_package = {
        "README.md": "# Habit Tracker Application Blueprint\n\nThis package contains the complete blueprint for developing a habit tracking web application.",
        "PRD.md": "# Product Requirements Document\n\n## Executive Summary\nA web application that helps users track daily habits and visualize progress.",
        "TechDesignDoc.md": "# Technical Design Document\n\n## Tech Stack\n- Frontend: React 18.2.0 with Vite\n- Backend: Node.js 18.x with Express 4.18.2",
        "NOTES.md": "# Implementation Notes\n\n## Step 1: Project Setup\n```bash\nmkdir -p habit-tracker/src/components\ncd habit-tracker\nnpm init -y\n```",
        "TESTING_PLAN.md": "# Testing Plan\n\n## Testing Frameworks\n- Frontend: Jest with React Testing Library\n- Backend: Jest with Supertest"
    }
    
    # Display the final package
    ui.display_final_package(mock_final_package)

if __name__ == "__main__":
    main()
