from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langchain_core.messages import SystemMessage, HumanMessage

from reasoner_prompt import REASONER_PROMPT
from formatter_prompt import FORMATTER_PROMPT  
from agent_builder import AgentBuilder
from constants import KEEP_LAST_N

class Workflow_builder:

    def __init__(self, agent_framework):
        
        self.agent_framework = agent_framework
        self.react_workflow = None

        print("Workflow building started...")
        self.workflow()
        print("Workflow built successfully!!!")

    # create the stateGraph
    def workflow(self):
        builder = StateGraph(MessagesState)
        builder.add_node("node_1", self.node_assistant)
        builder.add_node("tools", ToolNode(self.agent_framework.tools_list))
        builder.add_node("formatter", self.format_node)

        # add edges 
        builder.add_edge(START, "node_1")
        builder.add_conditional_edges("node_1", tools_condition,
                                      {
                                          "tools": "tools",
                                          "__end__": "formatter"
                                      })
        builder.add_edge("tools", "node_1")
        builder.add_edge("formatter", END)

        # compile the workflow with the memory
        #self.react_workflow = builder.compile(checkpointer=self.agent_framework.memory)
        self.react_workflow = builder.compile()

        # 
        print("built node")
        self.visualise_workflow()

    # define node working
    def node_assistant(self, State: MessagesState):

        # do something with the input state
        print("node_assistant running")
        sys_msg = SystemMessage(content= REASONER_PROMPT)   

        # Stage 0: Applying Message Summarization
        past_summary_and_recent_messages = self.summarize_old_tool_outputs(State["messages"], KEEP_LAST_N)

        # Stage 1: Reasoning/Tool-use Agent
        response = self.agent_framework.llm_with_tools.invoke([sys_msg]+ 
                                                               past_summary_and_recent_messages
                                                            )
        print("node assistant finished")                                                
        return {"messages": [response]}

    def format_node(self, state: MessagesState):
        print("format node running")
        query = "Query:" + state["messages"][0].content
        last_answer = "Final Answer:" + state["messages"][-1].content
        print(f"{FORMATTER_PROMPT}: **Answer** {last_answer}")

        response = self.agent_framework.llm.invoke([
            SystemMessage(content=FORMATTER_PROMPT),
            HumanMessage(content=query),
            HumanMessage(content=last_answer),
        ])
        print("format node finished")
        return {"messages": [response]}

    def format_answer(self, raw_output: str) -> str:
        print(f"In the static formatter.")
        import re 
        import json
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if not match:
            return raw_output.strip()
        try:
            items = json.loads(match.group())
            return ", ".join(str(item).strip() for item in items)  # comma + space, per your rule
        except json.JSONDecodeError:
            return raw_output.strip()  

    def visualise_workflow(self):
        display(Image(self.react_workflow.get_graph().draw_mermaid_png()))

    def summarize_old_tool_outputs(self, messages: list, keep_last_n: int = 4) -> list:
        """Once messages exceed a threshold, collapse older ToolMessages into
        one summary message, keeping only the most recent N raw."""
        if len(messages) <= keep_last_n:
            return messages

        old, recent = messages[:-keep_last_n], messages[-keep_last_n:]
        old_tool_content = "\n".join(
            m.content for m in old if hasattr(m, "content") and isinstance(m.content, str)
        )
        summary_llm = self.agent_framework.llm  # a plain, tool-free call
        summary = summary_llm.invoke([
            SystemMessage(content="Summarize the key facts found so far, concisely, preserving any specific numbers/names/dates."),
            HumanMessage(content=old_tool_content)
        ])
        return [SystemMessage(content=f"[Earlier findings summary]: {summary.content}")] + recent


# agent = AgentBuilder()
# task = Workflow_builder(agent)
# graph = task.react_workflow

if __name__ == "__main__":

    agent = AgentBuilder()
    task = Workflow_builder(agent)
#query = """In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"""
#query = """Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?"""
# query = """Given this table defining * on the set S = {a, b, c, d, e}

#         |*|a|b|c|d|e|
#         |---|---|---|---|---|---|
#         |a|a|b|c|b|d|
#         |b|b|c|a|e|c|
#         |c|c|a|b|b|a|
#         |d|b|e|b|e|d|
#         |e|d|b|a|d|c|

#         provide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order."
#         """
    query = """Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."""

    messages = {"messages": [query]}
    config = agent.config
    response = task.react_workflow.invoke(messages, config)

    print(response["messages"][-1].content)
    print(task.format_answer(response["messages"][-1].content))