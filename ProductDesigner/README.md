# Deep Planning LangGraph

A multi-agent system that transforms a user's initial idea into a developer-ready blueprint. This system uses LangGraph to orchestrate specialized agents that collaborate to create comprehensive project documentation.

## System Overview

This LangGraph is designed as a multi-agent system where each agent has a distinct role, collaborating to transform a user's initial idea into a developer-ready blueprint. The process is iterative, user-centric, and grounded through web search and human-in-the-loop validation, ensuring the final output is robust, complete, and unambiguous.

## Components

1. **Product Interviewer Agent**: Acts as a combined Product Manager and Technical Lead
2. **PRD Generator Agent**: Transforms structured Q&A into a formal Product Requirements Document
3. **Technical Design Generator Agent**: Creates detailed Technical Design Documents
4. **User Review & Feedback**: Human-in-the-loop validation checkpoint
5. **Technical Manager Agent**: Synthesizes documents into implementation instructions
6. **Test Developer Agent**: Generates comprehensive testing plans
7. **Final Assembler**: Bundles all artifacts into a complete project blueprint

## Usage

### Web Interface (Recommended)

1. Start the web application:
```bash
python run_web_app.py
```

2. Open your browser to `http://localhost:5000`

3. Enter your project idea and follow the interactive planning process:
   - **Product Interview**: AI conducts structured discovery through 13 key questions
   - **Document Generation**: Creates PRD and Technical Design documents
   - **Implementation Guide**: Provides step-by-step development instructions
   - **Testing Plan**: Generates comprehensive testing strategy
   - **Final Blueprint**: Downloads complete project package

### Command Line Interface

1. Run the main demo:
```bash
python main.py
```

2. The system will:
   - Create a mock product idea
   - Run through the complete multi-agent workflow
   - Generate all required documents
   - Save the final blueprint package

3. Check the generated files in the output directory for your complete project blueprint.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
