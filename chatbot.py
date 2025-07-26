from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# OpenRouter configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    max_tokens=1000
)

class State(dict):
    message: List[Dict[str, str]]  # Type hint for IDE, linters, etc.

    def __init__(self):
        super().__init__()
        self["message"] = []



def chatbot(state: State) -> State:
    # Send the ENTIRE message history to the LLM
    response = llm.invoke(state["message"])
    print(state["message"])  # Debugging line to print the message history
    state["message"].append({"role": "assistant", "content": response.content})
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

conversation_state = State()

def stream_chatbot(user_message: str):
    global conversation_state

    # Add user message to state
    conversation_state["message"].append({"role": "user", "content": user_message})

    # Stream output through LangGraph
    for event in graph.stream(conversation_state):
        for value in event.values():
            print("Assistant:", value["message"][-1]["content"])

if __name__ == "__main__":
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            stream_chatbot(user_input)
        except Exception as e:
            print("Error:", e)
            break