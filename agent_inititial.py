import logging
import json
import os
from typing import Annotated, TypedDict, List, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from prompts import SYSTEM_PROMPT, NARRATION_PLAN
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, langchain, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation
from qdrant_client import QdrantClient
from openai import OpenAI

logger = logging.getLogger("basic-agent")

load_dotenv()

# pip install langchain langgraph openai qdrant-client

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph

# Initialize OpenAI and Qdrant clients
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# ------------- RAG helpers -------------
def embed_query(text: str) -> List[float]:
    resp = oai.embeddings.create(model=os.getenv("OPENAI_EMBED_MODEL"), input=[text])
    return resp.data[0].embedding

def retrieve(query: str, k: int = 2) -> List[Dict]:
    vec = embed_query(query)
    res = qdrant.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION"),
        query=vec,              # the dense vector
        limit=k,
        with_payload=True,      # same as before
        score_threshold=0.2,  # optional, server-side filter
    )
    hits = res.points
    print("hits", hits)
    out = []
    for h in hits:
        payload = h.payload or {}
        print("payload", payload)
        chunk = payload.get(os.getenv("TEXT_PAYLOAD_KEY")) or payload.get("chunk") or payload.get("content") or ""
        if not chunk:
            continue
        if h.score is not None and h.score >= 0.2:
            out.append({
                "text": chunk
            })
    print("\n \n out : ", out)
    return out

# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# -----------------------------
# 2. Create LLM + prompt template
@tool
def query_vector_db(query: str) -> str:
    """Query the vector database for additional context"""
    try:
        results = retrieve(query, k=3)  # Get top 3 results
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Combine the retrieved chunks into a single context string
        context_chunks = []
        for i, result in enumerate(results, 1):
            context_chunks.append(f"Context {i}: {result['text']}")
        
        return "\n\n".join(context_chunks)
    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        return "Sorry, I encountered an error while searching the knowledge base. Please try again."

# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([query_vector_db])
tools_by_name = {"query_vector_db": query_vector_db}

# -----------------------------
# 3. Define nodes
# -----------------------------
def start_node(state: AgentState) -> AgentState:
    # Initialize with empty messages - system messages are handled by prompt template
    return {"messages": []}


def call_llm(state: AgentState) -> AgentState:
    # Check if user wants to exit
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_input = last_message.content.lower().strip()
            if user_input in ["exit", "quit", "bye", "goodbye", "end", "stop"]:
                # Return a goodbye message and signal to exit
                goodbye_message = "Thank you for the knowledge transfer session! I hope you found the information about the ABC Pvt. Ltd. HR policies helpful. Goodbye!"
                return {"messages": [AIMessage(content=goodbye_message)]}
    
    # Build the conversation with system prompt and narration plan
    conversation = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"NARRATION_PLAN:\n{NARRATION_PLAN}")
    ] + state["messages"]
    print(state["messages"])
    # Get response from LLM
    response = llm_with_tools.invoke(conversation)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_id = tool_call["id"]
        tool = tools_by_name[tool_name]

        state["messages"].append(response)
        print(f"Tool input: {tool_input}")
        tool_response = tool.invoke(tool_input)
        print(f"Tool response: {tool_response}")

        state["messages"].append(
            ToolMessage(
                content=json.dumps(tool_response),
                tool_call_id=tool_id,
            )
        )

        response = llm.invoke(state["messages"])
    
    
    return {"messages": [AIMessage(content=response.content)]}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# a simple StateGraph with a single GPT-4o node
def create_graph() -> StateGraph:

    workflow = StateGraph(AgentState)

    workflow.add_node("start", start_node)
    workflow.add_node("llm_node", call_llm)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "llm_node")
    # For LiveKit agent, we don't need complex routing - just process each message
    
    return workflow.compile()


async def entrypoint(ctx: JobContext):
    graph = create_graph()

    agent = Agent(
        instructions="You are an AI agent providing knowledge transfer about ABC Pvt. Ltd. HR policies. Start by introducing yourself and the HR policy overview. If the user says 'exit', 'quit', 'bye', 'goodbye', 'end', or 'stop', respond with a polite goodbye and the session will end.",
        llm=langchain.LLMAdapter(graph),
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        # any combination of STT, LLM, TTS, or realtime API can be used
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # to use Krisp background voice cancellation, install livekit-plugins-noise-cancellation
            # and `from livekit.plugins import noise_cancellation`
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    # Start the knowledge transfer session
    await session.generate_reply(instructions="Welcome! I'm here to provide knowledge transfer about ABC Pvt. Ltd. HR policies. Let me start by giving you an overview of our HR framework. You can say 'exit' anytime to end this session.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
