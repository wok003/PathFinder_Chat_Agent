import httpx, asyncio, logging
import trafilatura
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from orchestrator.constants import TOP_K_WEB_PAGES, CHUNK_OVERLAP, CHUNK_SIZE, MAX_RESULTS
from langsmith import traceable
from common.logger_config import setup_logging

## TODO:: 
# 0. Rigorously unit test the functionalities.
# 1a. Create Dataset to iteratively improve the toolbox
# 1b. Improve Agent's Toolbox : Use RAG cookbook, Contextual Embedding, and Knowledge Graphs 
# 1c. Improve ToolBox Descriptions
  
# --- API KEYS ---
load_dotenv()

# --- Basic Agent Definition ---
class Toolbox:
    "Agent toolbox contains:"
    "(1) string reversal capability,"
    "(2) web search, open web page, and extract relevant contents capability."

    def __init__(self, max_results):
        """Initializes search, split, and retrieval tools"""
        self.tavily_search = TavilySearch(max_results=max_results)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # --- Add Logger
        self.logger = logging.getLogger(__name__)

    @traceable(run_type="tool")
    def reverse_string(self, text: str) -> str:
        """Reverses a string character by character. Use this for any task
        involving reversed text, mirrored sentences, or character-level
        string manipulation - do NOT attempt this via reasoning alone."""

        return text[::-1]
    
    @traceable
    # splits, embeds, and finds the best result
    def retrieve_top_chunks(self, full_text: str, query: str, top_k: int = 3) -> str:
        """Fetch a page, then return only the chunks most relevant to the query,
        instead of the whole page or a blind character cutoff."""

        chunks = self.splitter.split_text(full_text)
        if not chunks:
            self.logger.critical("No extractable content found.")
            return "No extractable content found."

        chunk_embeddings = self.embedder.embed_documents(chunks)
        query_embedding = self.embedder.embed_query(query)

        # cosine similarity, ranked
        sims = [np.dot(query_embedding, c) / (np.linalg.norm(query_embedding) * np.linalg.norm(c))
                for c in chunk_embeddings]
        self.logger.debug(f"similarity calculations: {sims}")
        top_indices = np.argsort(sims)[-top_k:][::-1]

        return "\n---\n".join(chunks[i] for i in top_indices)
    
    @traceable
    async def fetch_one(self, url, client):
        try:
            response = await client.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

        self.logger.debug(f"url: {url}, text: {response.text}")
        text = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        self.logger.debug(f"url: {url}, cleaned text: {text}")

        return text  # already clean, no manual tag-stripping needed
    
    @traceable(run_type="retriever")
    async def fetch_and_retrieve_multi(self, urls: list[str], query: str, top_k_chunks: int = 3) -> str:
        """"""
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*[self.fetch_one(u, client) for u in urls])

        combined_text = "\n\n".join(r for r in results if r)  # drop failures, keep successes
        self.logger.debug(f"combined webpages: {combined_text}")

        if not combined_text:
            self.logger.critical(f"Could not retrieve content from any source.: {combined_text}")
            return "Could not retrieve content from any source."
        
        return self.retrieve_top_chunks(combined_text, query, top_k_chunks)  # your existing retriever

    # tools access: web search
    @traceable(metadata={"search_provider": "Tavily"})
    def web_search_raw(self, query: str):
        """Searches query/facts on the web and returns the top 3 pages with content

        Args:
            query: unknown fact. 
        """
        try:
            data = self.tavily_search.invoke({"query": query})
            search_docs = data.get("results", data)
            self.logger.info(f"Searched Successfully: query: {query}, returned results: {search_docs}")
            return search_docs
        
        except Exception as e:
            self.logger.critical(f"Tavily Web Search unavailable:: \n query: {query}")
            # TODO: Not implemented as not a big issue.
            # create exponential search loop
            # fallback plan
            return "WEB SEARCH FAILED"

    # main tool 
    @traceable(run_type="tool")
    def search_and_retrieve(self, query: str) -> str:
        """Search the web and return the most relevant content for the query,
        already fetched and filtered to the top few relevant passages."""

        self.logger.info("web search and retrieve tool call")
        search_results = self.web_search_raw(query)  # your existing Tavily call
        url = [search_results[i]["url"] for i in range(len(search_results))]
        self.logger.debug(f"Extracted URLs: {url}")
        
        return asyncio.run(self.fetch_and_retrieve_multi(url, query, TOP_K_WEB_PAGES))
       
class AgentBuilder:
    """builds a simple agent with basic tools"""
    @traceable
    def __init__(self):
        """agent creation"""
        print("Agent Building Started...")

        # foundational agentic capability require 3 things
        # -> reasoning instrument
        # -> memory instrument 
        # -> query instrument

        # reasoner instrument
        self.llm_with_tools = None
        self.llm = None

        # memory instrument
        self.memory = None 
        self.config = None 

        # query instrument
        self.tools_list = None
        self.tool_box = Toolbox(max_results=MAX_RESULTS)

        # initialize hte logger
        self.logger = logging.getLogger(__name__)
        
        # 1. build the chat llm
        self.agent_framework()

        # 2. Initialize the memory block
        self.agent_memory()
        self.logger.info("Agent Built Successfully with memory blocks!!!")
        
    @traceable
    def agent_memory(self):
        """Creates memory to log agent state info at each node in the workflow."""
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}

    @traceable
    def agent_tools(self):
        """ Tools for the agent"""
        self.tools_list = [self.tool_box.search_and_retrieve, self.tool_box.reverse_string]

    @traceable(run_type="llm", metadata={"ls_model_name": "qwen3:8b", "ls_provider": "Ollama"})
    def agent_mind(self):
        """Reasoner Definition"""
        self.llm = ChatOllama(
            model = "qwen3:8b",
            temperature = 0
        )       

    @traceable(run_type="chain")    
    def agent_framework(self):
        """Calls reasoner and the toolbox, then binds tools to the reasoner.
            - binding allows automatic injection of the tool related schema and prompt.
            - generates json schema required for tool calling."""
        self.agent_mind()
        self.logger.info("reasoner added!!")

        self.agent_tools()
        self.logger.info("tools connected to the reasoner!!")

        self.llm_with_tools = self.llm.bind_tools(self.tools_list) 
        self.logger.info("llm with tool calling capability available as 'llm_with_tool_calls'")

if __name__ == "__main__":

    info_tool = Toolbox(MAX_RESULTS)
    query = "Which is the Current Top Ranking Team in ICC Test CRICKET Mens Team Ranking ?"
    summary=info_tool.search_and_retrieve(query=query, langsmith_extra={"metadata":{"runtime_metadata": "foo"}})
    print("WEB ANSWER")
    print(summary)
    print("----------------------------------------")

    ## testing tracing in Agent Builder
    my_agent = AgentBuilder()
    response = my_agent.llm_with_tools.invoke("Top team ranking in FIFA WorldCup ?")
    print("----------------------------")
    print(response)
    print("----------------------------")

# i mean if the query is not specific then the web pages might get fetched 
# from a alltogether different topic like I ASKED something about cricket 
# but it fetched something from banking. SOLUTIONS::

# SOLUTION 1: PROMPT DESIGN - **better first evaluate with Model Related Query**
# When forming a search query, always include the specific named entities,
# domain, or subject from the original question (e.g. sport, industry, person,
# event name) - do not paraphrase them away. A vague query risks returning
# results from a completely unrelated domain.