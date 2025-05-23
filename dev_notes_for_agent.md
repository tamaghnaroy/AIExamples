# Generate NOTES.md for AI IDE Agent

**Goal:** To generate a structured `NOTES.md` file that summarizes key information and provides a high-level implementation plan based on the provided Product Requirements Document (PRD) and Technical Design Document (Tech Design Doc). This `NOTES.md` file will serve as the primary guide for an AI IDE agent (like GitHub Copilot Chat or Cursor) during the initial code generation phase.

**AI Role:** Act as a meticulous Technical Project Manager. Your task is to thoroughly analyze the referenced PRD and Tech Design Doc files and synthesize them into a clear, actionable `NOTES.md` file formatted for an AI agent.

**Reference Documents:**

*   **Product Requirements Document (PRD):** Please read the attached/provided PRD file (`[Specify PRD Filename Here, e.g., PRD-MVP.md]`). This contains the *what* – requirements, user stories, features, scope.
*   **Technical Design Document (Tech Design Doc):** Please read the attached/provided Tech Design Doc file (`[Specify Tech Design Doc Filename Here, e.g., Tech-Design-MVP.md]`). This contains the *how* – tech stack, architecture, data flow, implementation approach.

**Instructions for AI:**

1.  **Analyze Thoroughly:** Carefully read and understand *both* the PRD and Tech Design Doc in their entirety.
2.  **Synthesize Information:** Extract and combine relevant information from both documents.
3.  **Structure Output:** Generate a markdown file named `NOTES.md` containing the following sections:
    *   **`## Project Overview`**:
        *   `Product Name`: (From PRD)
        *   `Core Purpose`: (Concise summary based on PRD)
        *   `MVP Goal`: (Primary goal for the MVP, from PRD)
        *   `Target Audience`: (From PRD)
    *   **`## Technical Specifications (from Tech Design Doc)`**:
        *   `Platform`:
        *   `Tech Stack (Frontend)`:
        *   `Tech Stack (Backend/Core)`:
        *   `Key Libraries/APIs`:
        *   `Architecture Overview`: (Brief description or key components)
        *   `Data Handling Notes`: (Key points on privacy/storage)
        *   `Error Handling Approach`: (Brief summary)
    *   **`## Core MVP Features & Implementation Plan (from PRD & Tech Design Doc)`**:
        *   For *each* core MVP feature listed in the PRD:
            *   Create a sub-section: `### Feature: [Feature Name]`
            *   `Description`: (From PRD)
            *   `Key Acceptance Criteria/User Story`: (Link to main user story if applicable, or list key criteria)
            *   `Technical Implementation Notes`: (Summarize approach from Tech Design Doc, mention key components/files involved if specified)
            *   `Agent Implementation Steps (Suggested)`: (Provide a *high-level*, logical checklist for the agent, e.g., "1. Create function X in file Y.py", "2. Add route Z in main app file", "3. Implement UI element based on UI concept")
    *   **`## UI/UX Concept (from PRD)`**:
        *   Brief description of the look, feel, or key elements.
    *   **`## Out of Scope for MVP (from PRD)`**:
        *   List features explicitly excluded.
    *   **`## Key Agent Instructions`**:
        *   "Agent: Please generate the MVP codebase based on the details above."
        *   "Prioritize implementing the features exactly as specified in the 'Core MVP Features' section."
        *   "Strictly adhere to the 'Technical Specifications' regarding platform, stack, and architecture."
        *   "Refer to the full PRD (`[PRD Filename]`) and Tech Design Doc (`[Tech Design Doc Filename]`) files in the project root for complete details if needed."
        *   "Create files and directory structures as logically required by the Tech Design Doc and implementation plan."
        *   "Add comments to explain complex logic."

4.  **Clarity and Conciseness:** Ensure the generated `NOTES.md` is clear, well-organized, and provides actionable guidance for the AI agent. Use markdown formatting effectively.
5.  **Filename Placeholders:** Remember to include the actual filenames of the PRD and Tech Design Doc in the "Key Agent Instructions" section where indicated.

**Output:**

*   The complete content for the `NOTES.md` file, formatted in markdown.

---

**Execution Request:**

Please generate the `NOTES.md` content by analyzing the referenced PRD (`[Specify PRD Filename Here]`) and Tech Design Doc (`[Specify Tech Design Doc Filename Here]`) files according to the instructions above.
