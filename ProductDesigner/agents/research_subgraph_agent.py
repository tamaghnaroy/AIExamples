from typing import Dict, Any, List, Optional, Union
import sys
import os
import logging
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END
try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

# Jupyter notebook compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
    JUPYTER_COMPATIBLE = True
except ImportError:
    JUPYTER_COMPATIBLE = False

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SearchProtocol(str, Enum):
    """Enum for different search protocols."""
    OPENAI_WEB_SEARCH = "openai_web_search"
    FIRECRAWL = "firecrawl"
    # Future protocols can be added here
    # SERP_API = "serp_api"
    # GOOGLE_SEARCH = "google_search"

# Individual search result from a specific protocol
class SearchResult(BaseModel):
    """Individual search result from a specific protocol."""
    protocol: SearchProtocol
    query: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

# List of search results per query and the the learning derived from it
class Learning(BaseModel):
    """Unified learning structure combining results from multiple search protocols."""
    query: str
    search_results: List[SearchResult] = Field(default_factory=list)
    combined_content: str = ""
    synthesis_summary: Optional[str] = None
    confidence_score: float = 0.0
    
    def add_search_result(self, result: SearchResult):
        """Add a search result to this learning."""
        self.search_results.append(result)
        self._update_combined_content()
    
    def _update_combined_content(self):
        """Update the combined content from all search results."""
        content_parts = []
        for result in self.search_results:
            if result.success and result.content:
                content_parts.append(f"[{result.protocol.value}] {result.content}")
        self.combined_content = "\n\n---\n\n".join(content_parts)

    def __str__(self):
        return f"Query: {self.query}\nCombined Content: {self.combined_content}"

# Hold the list of search queries to be executed
class Websearch(BaseModel):
    """Pydantic model for web search queries."""
    search_queries: List[str]

# State for the research subgraph workflow
class ResearchState(BaseModel):
    """State for the research subgraph workflow."""
    depth: int = 1
    max_depth: int = 5

    initial_idea: str = Field(description="Initial idea for the research")

    current_queries: List[str] = Field(default_factory=list, description="Current web search queries at current depth level")
    learnings: List[Learning] = Field(default_factory=list, description="Learnings from web search - cummulative")
    
    knowledge_base: str = Field(default="", description="Knowledge base from the list of learnings")
    research_completed: bool = False
    
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# ----------------------------------------------
# Functions for parallel websearch for queries
# ----------------------------------------------

async def _execute_openai_search(query: str) -> SearchResult:
    """Execute OpenAI web search for a single query."""
    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await client.responses.create(
            model="gpt-4.1",
            input="Use web_search to search for information for each of the following topics - \n" + query,
            tools=[
                {
                    "type": "web_search"
                }
            ]
        )
        
        content = response.output[1].content[0].text
        
        return SearchResult(
            protocol=SearchProtocol.OPENAI_WEB_SEARCH,
            query=query,
            content=content,
            metadata={"model": "gpt-4.1"},
            success=True
        )
        
    except Exception as e:
        logging.error(f"OpenAI search failed for '{query}': {e}")
        return SearchResult(
            protocol=SearchProtocol.OPENAI_WEB_SEARCH,
            query=query,
            content="",
            success=False,
            error_message=str(e)
        )

async def _execute_firecrawl_search(query: str) -> SearchResult:
    """Execute Firecrawl search for a single query."""
    try:
        if FirecrawlApp is None:
            raise Exception("Firecrawl not available")
            
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
        # Enhanced Firecrawl API call with scraping options for better content extraction
        try:
            # Try using the newer API with scrape_options
            from firecrawl import ScrapeOptions
            scrape_options = ScrapeOptions(
                formats=["markdown"],
                onlyMainContent=True,
                includeTags=["title", "meta", "h1", "h2", "h3", "h4", "h5", "h6", "p", "article", "section"],
                excludeTags=["nav", "footer", "aside", "header", "script", "style"]
            )
            search_result = app.search(query, limit=5, scrape_options=scrape_options)
        except (ImportError, TypeError) as e:
            # Fallback to basic search if ScrapeOptions not available or API changed
            logging.info(f"Using basic search due to: {e}")
            search_result = app.search(query)
        
        # Debug: Log the response structure
        logging.info(f"Firecrawl response type: {type(search_result)}")
        logging.info(f"Firecrawl response attributes: {dir(search_result)}")
        
        # Extract and combine content
        combined_content = ""
        metadata = {"results_count": 0}
        
        # Handle SearchResponse object from Firecrawl
        data_items = []
        
        # Check if it's a SearchResponse object and try to access its data
        if hasattr(search_result, 'data'):
            data_items = search_result.data
            logging.info(f"Found data attribute with {len(data_items) if data_items else 0} items")
        elif hasattr(search_result, 'results'):
            data_items = search_result.results
            logging.info(f"Found results attribute with {len(data_items) if data_items else 0} items")
        elif hasattr(search_result, '__dict__'):
            # Try to convert the object to dict and look for data
            obj_dict = search_result.__dict__
            logging.info(f"SearchResponse object dict keys: {list(obj_dict.keys())}")
            if 'data' in obj_dict:
                data_items = obj_dict['data']
            elif 'results' in obj_dict:
                data_items = obj_dict['results']
        elif isinstance(search_result, dict):
            # Fallback to dict handling
            if 'data' in search_result:
                data_items = search_result['data']
            elif 'results' in search_result:
                data_items = search_result['results']
        elif isinstance(search_result, list):
            # If it's directly a list of results
            data_items = search_result
        
        if data_items:
            metadata["results_count"] = len(data_items)
            for item in data_items:
                # Handle different item structures
                if isinstance(item, dict):
                    # Try different ways to extract title, URL, and content
                    title = (
                        item.get('title') or 
                        item.get('metadata', {}).get('title') or 
                        item.get('name') or 
                        'No title'
                    )
                    
                    url = (
                        item.get('url') or 
                        item.get('metadata', {}).get('sourceURL') or 
                        item.get('metadata', {}).get('url') or 
                        item.get('link') or 
                        'No URL'
                    )
                    
                    content = (
                        item.get('markdown') or 
                        item.get('content') or 
                        item.get('text') or 
                        item.get('description') or 
                        item.get('snippet') or 
                        str(item)  # Fallback to string representation
                    )
                    
                    combined_content += f"\n--- {title} ({url}) ---\n{content}\n==============\n"
                else:
                    # If item is not a dict, just convert to string
                    combined_content += f"\n--- Result ---\n{str(item)}\n==============\n"
        
        # Add delay for rate limiting
        await asyncio.sleep(0.5)
        
        return SearchResult(
            protocol=SearchProtocol.FIRECRAWL,
            query=query,
            content=combined_content if combined_content else f"No results found for: {query}",
            metadata=metadata,
            success=True
        )
        
    except Exception as e:
        logging.error(f"Firecrawl search failed for '{query}': {e}")
        return SearchResult(
            protocol=SearchProtocol.FIRECRAWL,
            query=query,
            content="",
            success=False,
            error_message=str(e)
        )

async def _execute_parallel_searches_for_queries(queries: List[str], enabled_protocols: List[SearchProtocol]) -> List[Learning]:
    """Execute all queries across all protocols in parallel simultaneously."""
    # Create all search tasks upfront - every query x every protocol
    all_search_tasks = []
    task_metadata = []  # Track which task corresponds to which query/protocol
    logging.info("---1 ---")
    for query in queries:
        logging.info(f"--- {query} ---")
        
        if SearchProtocol.OPENAI_WEB_SEARCH in enabled_protocols:
            logging.info(f"--- {SearchProtocol.OPENAI_WEB_SEARCH.value} ---")
            task = _execute_openai_search(query)
            all_search_tasks.append(task)
            task_metadata.append((query, SearchProtocol.OPENAI_WEB_SEARCH))
        
        if SearchProtocol.FIRECRAWL in enabled_protocols:
            logging.info(f"--- {SearchProtocol.FIRECRAWL.value} ---")
            task = _execute_firecrawl_search(query)
            all_search_tasks.append(task)
            task_metadata.append((query, SearchProtocol.FIRECRAWL))
    logging.info("--- 2 ---")
    logging.info(f"Executing {len(all_search_tasks)} search tasks in parallel across {len(queries)} queries and {len(enabled_protocols)} protocols")
    
    # Execute ALL tasks in parallel simultaneously
    if all_search_tasks:
        all_results = await asyncio.gather(*all_search_tasks, return_exceptions=True)
    else:
        all_results = []
    
    # Group results by query to create Learning objects
    learning_dict = {query: Learning(query=query) for query in queries}
    
    # Process all results and group them by query
    for result, (query, protocol) in zip(all_results, task_metadata):
        if isinstance(result, SearchResult):
            learning_dict[query].add_search_result(result)
            logging.debug(f"Added {protocol.value} result for query: {query}")
        elif isinstance(result, Exception):
            logging.error(f"Search task failed for query '{query}' with protocol '{protocol.value}': {result}")
            # Add failed result to maintain consistency
            failed_result = SearchResult(
                protocol=protocol,
                query=query,
                content="",
                success=False,
                error_message=str(result)
            )
            learning_dict[query].add_search_result(failed_result) 
    
    # Convert to list maintaining original query order
    learnings = [learning_dict[query] for query in queries]
    
    logging.info(f"Completed parallel execution for {len(queries)} queries across {len(enabled_protocols)} protocols")
    return learnings

def _execute_searches_sequentially(queries: List[str], enabled_protocols: List[SearchProtocol]) -> List[Learning]:
        """Fallback method to execute searches sequentially when async is not available."""
        logging.info("Executing searches sequentially as fallback")
        learnings = []
        
        for query in queries:
            learning = Learning(query=query)
            
            # Execute each protocol sequentially
            if SearchProtocol.OPENAI_WEB_SEARCH in enabled_protocols:
                try:
                    result = asyncio.run(_execute_openai_search(query))
                    learning.add_search_result(result)
                except Exception as e:
                    logging.error(f"Sequential OpenAI search failed for '{query}': {e}")
            
            if SearchProtocol.FIRECRAWL in enabled_protocols:
                try:
                    result = asyncio.run(_execute_firecrawl_search(query))
                    learning.add_search_result(result)
                except Exception as e:
                    logging.error(f"Sequential Firecrawl search failed for '{query}': {e}")
            
            learnings.append(learning)
            logging.info(f"Completed sequential search for query: {query}")
        
        return learnings

def _run_parallel_searches_sync(queries: List[str], enabled_protocols: List[SearchProtocol]) -> List[Learning]:
    """Synchronous wrapper for websearch across protocols for a given set of queries"""
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in Jupyter or another environment with a running loop
            if JUPYTER_COMPATIBLE:
                # Use nest_asyncio to allow nested async calls
                return asyncio.run(_execute_parallel_searches_for_queries(queries, enabled_protocols))
            else:
                # Fallback: run synchronously
                logging.warning("Running in environment with active event loop but nest_asyncio not available. Running searches sequentially.")
                return _execute_searches_sequentially(queries, enabled_protocols)
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(_execute_parallel_searches_for_queries(queries, enabled_protocols))
            
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No event loop, create one
            return asyncio.run(_execute_parallel_searches_for_queries(queries, enabled_protocols))
        else:
            # Other runtime error, fall back to sequential
            logging.warning(f"Event loop error: {e}. Falling back to sequential execution.")
            return _execute_searches_sequentially(queries, enabled_protocols)
    except Exception as e:
        logging.error(f"Error in parallel search execution: {e}")
        return _execute_searches_sequentially(queries, enabled_protocols)

# -------------------------------------------
# Main Agent
# -------------------------------------------

class ResearchSubgraphAgent(BaseAgent):
    """
    Agent that executes a research subgraph based on initial user idea.
    Uses LangGraph for orchestration and supports parallel execution of multiple search protocols.
    """

    def __init__(self):
        """Initialize the Research Subgraph Agent."""
        super().__init__()
        self.max_depth = 5
        self.enabled_protocols = [SearchProtocol.OPENAI_WEB_SEARCH]
        
        # Add Firecrawl if available
        if FirecrawlApp is not None and os.getenv("FIRECRAWL_API_KEY"):
            self.enabled_protocols.append(SearchProtocol.FIRECRAWL)
            
        self.research_graph = self._build_research_graph()
    
    def _build_research_graph(self) -> StateGraph:
        """Build the LangGraph workflow for research execution."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("execute_parallel_search", self._execute_parallel_search_node)
        workflow.add_node("evaluate_learnings", self._evaluate_learnings_node)
        workflow.add_node("generate_knowledge_base", self._generate_knowledge_base_node)
        
        # Define the flow
        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "execute_parallel_search")
        workflow.add_edge("execute_parallel_search", "evaluate_learnings")
        
        # Conditional edge for iteration or completion
        workflow.add_conditional_edges(
            "evaluate_learnings",
            self._should_continue_research,
            {
                "continue": "execute_parallel_search",
                "finish": "generate_knowledge_base"
            }
        )
        workflow.add_edge("generate_knowledge_base", END)
        
        return workflow.compile()
        
    def _run_llm_step(self, state: Dict[str, Any], prompt_template_str: str, params: Dict[str, Any], 
                     fallback_response: str, output_parser=None, model_override: str = None, 
                     temperature_override: float = None) -> Any:
        """Helper to run a single LLM step using the base agent's service."""
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        full_params = {**state, **params}
        return self.llm_service.execute(prompt_template, full_params, fallback_response, 
                                      output_parser, model_override, temperature_override)
    
    # -------------------------
    # Node 1: Generate Queries
    # -------------------------

    def _generate_queries_node(self, state: ResearchState) -> ResearchState:
        """LangGraph node to generate search queries.
           Update the "current_queries" field in the state with the generated search queries.
        
        Args:
            state (ResearchState): Current state of the research subgraph workflow
        
        Returns:
            ResearchState: Updated state with generated search queries
        """
        try:
            if state.depth == 1:
                # Initial query generation
                websearch = self.generate_search_queries_agent(state.initial_idea)
                state.current_queries = websearch.search_queries
            else:
                # Generate additional queries based on current learnings
                if state.learnings:
                    learning_texts = [learning.combined_content for learning in state.learnings]
                    additional_websearch = self.iterative_search_expansion_agent(state.initial_idea, learning_texts)
                    state.current_queries = additional_websearch.search_queries
                else:
                    websearch = self.generate_search_queries_agent(state.initial_idea)
                    state.current_queries = websearch.search_queries
                
            logging.info(f"Generated {len(state.current_queries)} queries for depth {state.depth}")
            return state
            
        except Exception as e:
            logging.error(f"Error in generate_queries_node: {e}")
            state.error = str(e)
            return state
    
    def generate_search_queries_agent(self, initial_idea: str) -> Websearch:
        """
        Sub-agent to generate a list of search queries based on the initial product idea.
        Used by LangGraph nodes for query generation.
        
        Args:
            initial_idea (str): The initial product idea from the user
            
        Returns:
            Websearch: Pydantic object containing list of search queries
        """
        logging.info("Generating initial search queries...")
        
        parser = PydanticOutputParser(pydantic_object=Websearch)
        format_instructions = parser.get_format_instructions()

        template = (
            "You are a useful product research and gap analysis agent. Based on the initial product idea from the user below, you will generate a list of queries to research the product idea from scratch. "
            "Focus on the following: "
            "1. Existing solution similar to the user idea "
            "2. Typical Feature set for a product with similarity to the user idea "
            "3. Examples of typical User journey for a product with similarity to the user idea "
            "4. Typical tech stack for a product with similarity to the user idea "
            "5. Existing codebases in Github which can be leveraged for implementing the user idea "
            "6. Third party python packages or open source projects / packages that can be used for implementing the user idea "
            "7. Examples of UI / UX for a product with similarity to the user idea "
            "8. Main functional modules for the product "
            "9. Gap analysis vs. existing solutions "
            "10. Features that are not required for the MVP "
            "11. Features that are useful but not necessary "
            "12. Data Sources if relevant, API details "
            "Return a maximum of 30 queries, but feel free to return less if that is sufficient. Make sure each query is unique and not similar to each other: \n"
            "Initial Idea: {initial_idea}\n"
            "Answer Format Instructions: "
            "\n{format_instructions}"
        )

        params = {
            "initial_idea": initial_idea,
            "format_instructions": format_instructions,
        }

        websearch = self._run_llm_step(
            dict(), template, params, 
            "Could not generate the list of web search queries.", 
            output_parser=parser, 
            model_override="gpt-4.1", 
            temperature_override=0.1
        )
        
        logging.info(f"Generated {len(websearch.search_queries)} search queries")
        for i, query in enumerate(websearch.search_queries, 1):
            logging.info(f"{i}. {query}")
            
        return websearch

    def iterative_search_expansion_agent(self, initial_idea: str, current_learnings: List[str]) -> Websearch:
        """
        Sub-agent to evaluate current learnings and generate additional search queries if needed.
        
        Args:
            initial_idea (str): The initial product idea
            current_learnings (List[str]): Current accumulated learnings
            
        Returns:
            Websearch: Object containing additional search queries (empty if sufficient)
        """
        logging.info("Evaluating current learnings for additional search needs...")
        
        parser = PydanticOutputParser(pydantic_object=Websearch)
        format_instructions = parser.get_format_instructions()
        
        learning_text = "========\n".join(current_learnings)

        template = (
            "You are evaluating the work of a research agent who was tasked to extract information from the web about a product idea suggested by the user. "
            "Initial Idea: {initial_idea}\n"
            "Information extracted by the research agent: \n {topics}\n"
            "Evaluate if you have sufficient information that the client might need to decide about how to execute the product idea. If not, generate a list of additional queries to extract information from the web. "
            "If the information set is sufficient, return an empty list. "
            "Answer Format Instructions: "
            "\n{format_instructions}"
        )

        params = {
            "initial_idea": initial_idea,
            "format_instructions": format_instructions,
            "topics": learning_text,
        }

        websearch = self._run_llm_step(
            dict(), template, params,
            "Could not generate the list of web search queries.",
            output_parser=parser,
            model_override="gpt-4.1",
            temperature_override=0.1
        )
        
        if len(websearch.search_queries) == 0:
            logging.info("No additional queries needed - research is sufficient.")
        else:
            logging.info(f"Generated {len(websearch.search_queries)} additional queries")
            for i, query in enumerate(websearch.search_queries, 1):
                logging.info(f"Additional {i}. {query}")
                
        return websearch

    # -------------------------
    # Node 2: Execute Searches 
    # -------------------------
    
    def _execute_parallel_search_node(self, state: ResearchState) -> ResearchState:
        """LangGraph node to execute parallel searches across all enabled protocols."""
        try:
            if not state.current_queries:
                logging.info("No queries to execute")
                return state
                
            # Execute searches in parallel - Jupyter compatible approach
            new_learnings = _run_parallel_searches_sync(state.current_queries, self.enabled_protocols)
            state.learnings.extend(new_learnings)
            logging.info(f"Added {len(new_learnings)} new learnings")
                
            return state
            
        except Exception as e:
            logging.error(f"Error in execute_parallel_search_node: {e}")
            state.error = str(e)
            return state
    
    # -------------------------
    # Node 3: Conditional Edge
    # -------------------------

    def _evaluate_learnings_node(self, state: ResearchState) -> ResearchState:
        """LangGraph node to evaluate current learnings and decide on next steps."""
        try:
            state.depth += 1
            logging.info(f"Evaluating learnings at depth {state.depth}")
            logging.info(f"Total learnings so far: {len(state.learnings)}")
            
            return state
            
        except Exception as e:
            logging.error(f"Error in evaluate_learnings_node: {e}")
            state.error = str(e)
            return state
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Conditional function to determine if research should continue."""
        if state.error:
            return "finish"
        if state.depth > state.max_depth:
            return "finish"
        if not state.current_queries:  # No more queries generated
            return "finish"
        return "continue"
    
    # -----------------------------------------------
    # Node 4: Generate Knowledge Base - Final Node
    # ------------------------------------------------
    def _generate_knowledge_base_node(self, state: ResearchState) -> ResearchState:
        """LangGraph node to generate final knowledge base from all learnings."""
        try:
            learning_texts = [learning.combined_content for learning in state.learnings]
            state.knowledge_base = self._generate_knowledge_base(state.initial_idea, learning_texts)
            state.research_completed = True
            logging.info("Knowledge base generation completed")
            return state
        except Exception as e:
            logging.error(f"Error in generate_knowledge_base_node: {e}")
            state.error = str(e)
            return state

    def _generate_knowledge_base(self, initial_idea: str, all_learnings: List[str]) -> str:
        """
        Sub-agent to generate a comprehensive knowledge base from all research learnings.
        
        Args:
            initial_idea (str): The initial product idea
            all_learnings (List[str]): All accumulated learnings from research
            
        Returns:
            str: Comprehensive knowledge base summary
        """
        logging.info("Generating knowledge base from research learnings...")
        
        learning_text = "\n\n========\n\n".join(all_learnings)
        
        template = (
            "You are a knowledge synthesis agent. Based on the initial product idea and all the research findings below, "
            "create a comprehensive knowledge base that organizes and synthesizes the information. "
            "Structure your response with clear sections covering: "
            "1. Executive Summary "
            "2. Existing Solutions and Features we can leverage for user product idea"
            "3. Technical Architecture & Stack Recommendations "
            "4. Feature Analysis (Core MVP vs Nice-to-have) "
            "5. User Experience & Journey Insights for the user product idea and those already covered by existing solutions "
            "6. Implementation Resources (Libraries, APIs, Code Examples) "
            "7. Gap Analysis - what additional features do we need to implement to make the user product idea successful"
            "8. Development Recommendations for architecture, functional modules, data models, APIs  "
            "9. UI/UX recommendations for the user product idea"
            "\n"
            "Initial Idea: {initial_idea}\n"
            "Research Findings: \n{research_findings}\n"
            "\n"
            "Provide a well-structured, actionable knowledge base that will help in product development decisions."
        )

        params = {
            "initial_idea": initial_idea,
            "research_findings": learning_text,
        }

        knowledge_base = self._run_llm_step(
            dict(), template, params,
            "Could not generate knowledge base from research findings.",
            model_override="gpt-4.1",
            temperature_override=0.2
        )
        
        logging.info("Knowledge base generation completed")
        return knowledge_base

    # -------------------------------------
    # Node 7: Run Research Subgraph
    # -------------------------------------
    def run(self, state: GraphState) -> Dict[str, Any]:
        """
        Process the current state by executing the LangGraph-based research subgraph.
        
        Args:
            state (GraphState): The current state of the workflow
            
        Returns:
            Dict[str, Any]: Updated state with research results
        """
        try:
            initial_idea = state.initial_idea
            if not initial_idea:
                logging.error("No initial idea provided in state")
                return {"error": "No initial idea provided for research"}
            
            logging.info(f"Starting LangGraph-based research for idea: {initial_idea}")
            logging.info(f"Enabled protocols: {[p.value for p in self.enabled_protocols]}")
            
            # Create research state
            research_state = ResearchState(
                initial_idea=initial_idea,
                max_depth=self.max_depth
            )
            
            # Execute the LangGraph research workflow
            final_state = self.research_graph.invoke(research_state)
            
            # Extract results in the expected format
            research_results = {
                "research_method": "langgraph_parallel",
                "research_depth": final_state.depth - 1,  # Adjust for 1-based indexing
                "total_learnings": len(final_state.learnings),
                "learnings": [learning.dict() for learning in final_state.learnings],
                "knowledge_base": final_state.knowledge_base,
                "research_completed": final_state.research_completed,
                "enabled_protocols": [p.value for p in self.enabled_protocols]
            }
            
            if final_state.error:
                research_results["error"] = final_state.error
            
            # Return updated state
            return {
                "research_results": research_results,
                "research_knowledge_base": final_state.knowledge_base,
                "research_completed": final_state.research_completed,
                "research_learnings": final_state.learnings
            }   
        except Exception as e:
            logging.error(f"Error in ResearchSubgraphAgent.run: {e}")
            return {
                "error": str(e),
                "research_completed": False
            }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     