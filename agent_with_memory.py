from langgraph.graph import StateGraph,END,add_messages
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
import getpass
import os

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

class Mystate(TypedDict):
  messages: Annotated[list, add_messages]


graph = StateGraph(Mystate)

memory= MemorySaver()

def chat_node(state: Mystate):
  response = model.invoke(state["messages"])
  return {"messages": [response]}

graph.add_node("Chat", chat_node)
graph.set_entry_point("Chat")
graph.add_edge("Chat", END )

agent = graph.compile(checkpointer= memory)

configuration = {
    
                "configurable":{
                    "thread_id": 1
                }
}

ans = agent.invoke({"messages": [HumanMessage(content = "Write a LinkedIn post about how to improve productivity with Gen AI tools")]}, config = configuration)

ans2 = agent.invoke({"messages": [HumanMessage(content = "which topic did you write the post earlier on?")]}, config = configuration)

# printing the response in readable format. 

for response in [ans, ans2]:
    print("Conversation:")
    for msg in response["messages"]:
        if msg.type == "human":
            print(f"Human: {msg.content}\n")
        elif msg.type == "ai":
            print(f"AI: {msg.content}\n")
    print("-" * 40)

display(Image(agent.get_graph().draw_mermaid_png()))