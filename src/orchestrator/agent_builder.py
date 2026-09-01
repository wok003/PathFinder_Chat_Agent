import os
import httpx, asyncio
import trafilatura
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from orchestrator.constants import TOP_K_WEB_PAGES, CHUNK_OVERLAP, CHUNK_SIZE, MAX_RESULTS

# (Keep Constants as is)
# --- Constants ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Basic Agent Definition ---
class Toolbox:

    def __init__(self, max_results):
        self.tavily_search = TavilySearch(max_results=max_results, tavily_api_key=TAVILY_API_KEY)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def reverse_string(self, text: str) -> str:
        """Reverses a string character by character. Use this for any task
        involving reversed text, mirrored sentences, or character-level
        string manipulation - do NOT attempt this via reasoning alone."""
        return text[::-1]

    # splits, embeds, and finds the best result
    def retrieve_top_chunks(self, full_text: str, query: str, top_k: int = 3) -> str:
        """Fetch a page, then return only the chunks most relevant to the query,
        instead of the whole page or a blind character cutoff."""
        chunks = self.splitter.split_text(full_text)
        if not chunks:
            return "No extractable content found."

        chunk_embeddings = self.embedder.embed_documents(chunks)
        query_embedding = self.embedder.embed_query(query)

        # cosine similarity, ranked
        sims = [np.dot(query_embedding, c) / (np.linalg.norm(query_embedding) * np.linalg.norm(c))
                for c in chunk_embeddings]
        top_indices = np.argsort(sims)[-top_k:][::-1]

        return "\n---\n".join(chunks[i] for i in top_indices)

    async def fetch_one(self, url, client):
        try:
            response = await client.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

        text = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        return text  # already clean, no manual tag-stripping needed

    async def fetch_and_retrieve_multi(self, urls: list[str], query: str, top_k_chunks: int = 3) -> str:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*[self.fetch_one(u, client) for u in urls])
        combined_text = "\n\n".join(r for r in results if r)  # drop failures, keep successes
        if not combined_text:
            return "Could not retrieve content from any source."
        return self.retrieve_top_chunks(combined_text, query, top_k_chunks)  # your existing retriever

    # tools access: web search
    def web_search_raw(self, query: str):
        """Searches query/facts on the web and returns the top 3 pages with content

        Args:
            query: unknown fact. 
        """
        data = self.tavily_search.invoke({"query": query})
        search_docs = data.get("results", data)
        print(f"WEB SEARCH TOOL CALLED: {search_docs}")
        return search_docs
    
    # main tool 
    def search_and_retrieve(self, query: str) -> str:
        """Search the web and return the most relevant content for the query,
        already fetched and filtered to the top few relevant passages."""
        search_results = self.web_search_raw(query)  # your existing Tavily call
        print(search_results)
        url = [search_results[i]["url"] for i in range(len(search_results))]
        return asyncio.run(self.fetch_and_retrieve_multi(url, query, TOP_K_WEB_PAGES))
       
class AgentBuilder:

    def __init__(self):
        print("Agent Building Started...")
        self.llm_with_tools = None
        self.llm = None
        self.memory = None 
        self.config = None 
        self.tools_list = None
        self.tool_box = Toolbox(max_results=MAX_RESULTS)

        # 1. build the chat llm
        self.agent_framework()
        # 2. Initialize the memory block
        self.agent_memory()

        print("Agent Built Successfully!!!")

    def agent_memory(self):
        """Creates memory to log agent state info at each node."""
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}

    def agent_tools(self):
        """ Tools for the agent"""
        self.tools_list = [self.tool_box.search_and_retrieve, self.tool_box.reverse_string]
        
    def agent_mind(self):
        self.llm = ChatOllama(
            model = "qwen3:8b",
            temperature = 0
        )        
        
    def agent_framework(self):
        self.agent_mind()
        self.agent_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools_list) 

if __name__ == "__main__":

    info_tool = Toolbox(MAX_RESULTS)
    query = "Current Top Ten Test CRICKET Mens Team Ranking?"
    summary=info_tool.search_and_retrieve(query=query)
    print("WEB ANSWER")
    print(summary)
    print("----------------------------------------")

    # Testing the reverse string tool
    print(info_tool.reverse_string(query))

# i mean if the query is not specific then the web pages might get fetched 
# from a alltogether different topic like I ASKED something about cricket 
# but it fetched something from banking. SOLUTIONS::

# SOLUTION 1: PROMPT DESIGN - **better first evaluate with Model Related Query**
# When forming a search query, always include the specific named entities,
# domain, or subject from the original question (e.g. sport, industry, person,
# event name) - do not paraphrase them away. A vague query risks returning
# results from a completely unrelated domain.