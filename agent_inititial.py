import logging
import json
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from prompts import SYSTEM_PROMPT
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

logger = logging.getLogger("basic-agent")
roomContext = None

load_dotenv()

# pip install langchain langgraph openai


# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
class Slide(BaseModel):
    text: str
    slide_number: int
# -----------------------------
# 2. Create LLM + prompt template
@tool
def query_vector_db(query: str) -> str:
    """Query the vector database for additional context"""
    return """
ABC Pvt. Ltd. currently holds an estimated 2.5% of the global IT services market. Over the past five years, its market share has gradually increased from around 1.8% to 2.5%, reflecting steady growth in international client engagements and service offerings.""" 

# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([query_vector_db])
tools_by_name = {"query_vector_db": query_vector_db}

def get_narration_plan():
    with open("output/presentation_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    narration_plan = []
    for slide in data.get("slides", []):
        narration_plan.append(
            f"(Slide {slide['slide_number']})\nNarrator: {slide['narration_text']}\n"
        )
    return "\n".join(narration_plan)
def get_slide_json(slide_number: int):
    with open("output/presentation_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    slides = data.get("slides", [])
    return slides[int(slide_number)-1]["slide_json"] 
def send_data_to_room(data: dict):

    data_dict = {"type": "chat", "message": data, "sender": "agent"}
    json_data = json.dumps(data_dict)
    json_bytes = json_data.encode('utf-8')
    room = roomContext.room
    room.local_participant.publishData(json_bytes, reliable=True, topic="json")
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
    narration_plan = get_narration_plan()
    conversation = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"NARRATION_PLAN:\n{narration_plan}")
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
        
    structured_llm = llm.with_structured_output(Slide)
    response = structured_llm.invoke(state["messages"])
    response_text = response.text
    print("response", response_text)

   
    # send_data_to_room(get_slide_json(response.slide_number))
    return {"messages": [AIMessage(content=response.text)]}


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
    global roomContext
    roomContext = ctx
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
