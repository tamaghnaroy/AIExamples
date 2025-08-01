import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from ..utils.timeout_decorator import timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMService:
    """A modular service for handling all LLM interactions."""

    def __init__(self, api_key=None, model_name="gpt-4o", temperature=0.7):
        """
        Initializes the LLM service.

        Args:
            api_key (str, optional): The OpenAI API key. Defaults to env variable.
            model_name (str, optional): The model to use. Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model. Defaults to 0.7.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        self.output_parser = StrOutputParser()
        logging.info(f"LLMService initialized with model: {model_name}")

    def _create_chain(self, prompt_template: ChatPromptTemplate) -> Runnable:
        """Creates a LangChain runnable chain."""
        return prompt_template | self.llm | self.output_parser

    @timeout(30) # Using the same 30-second timeout
    def _invoke_chain_with_timeout(self, chain: Runnable, params: dict) -> str:
        """Invokes a chain with a timeout decorator."""
        try:
            return chain.invoke(params)
        except Exception as e:
            logging.error(f"LLM chain invocation failed: {e}")
            # The timeout decorator will raise TimeoutError, which is caught outside.
            raise

    def execute(self, prompt_template: ChatPromptTemplate, params: dict, fallback_response: str = "An error occurred while processing your request.",
    output_parser: str = None, model_override: str = None, temperature_override: float = None) -> str:
        """
        Executes an LLM call with a given prompt and parameters.

        Args:
            prompt_template (ChatPromptTemplate): The prompt template for the chain.
            params (dict): The parameters to pass to the prompt template.
            fallback_response (str, optional): The response to return on failure. Defaults to a generic error message.
            output_parser (str, optional): The output parser to use. Defaults to "llm_recommendation".

        Returns:
            str: The LLM's response or the fallback response.
        """
        if output_parser is not None:
            old_output_parser = self.output_parser
            self.output_parser = output_parser
        model = model_override if model_override is not None else self.model_name
        temperature = temperature_override if temperature_override is not None else self.temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        try:
            chain = self._create_chain(prompt_template)
            logging.info(f"Executing LLM chain with params: {params.keys()}")
            response = self._invoke_chain_with_timeout(chain, params)
            logging.info("LLM chain executed successfully.")
            return response
        except Exception as e:
            logging.error(f"LLM execution failed with error: {e}. Returning fallback response.")
            return fallback_response
        finally:
            if output_parser is not None:
                self.output_parser = old_output_parser
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, api_key=self.api_key)
