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
from livekit.plugins import openai, langchain, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation
from livekit import rtc
from livekit.agents import ModelSettings, Agent
from typing import Iterable

logger = logging.getLogger("basic-agent")
roomContext = None

load_dotenv()

# pip install langchain langgraph openai

roomContext = None
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
the food policy in keyvalue includes 3 free meals a week and one buffet dinner a month""" 

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
    print(f"narration_plan : {narration_plan}")
    return "\n".join(narration_plan)

def get_slide_json(slide_number: int):
    with open("output/presentation_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    slides = data.get("slides", [])
    slide =  slides[int(slide_number)-1]["slide_json"] 
    return slide


def send_data_to_room(slide_number: int):

    data_dict = {"type": "chat", "message": get_slide_json(slide_number), "sender": "agent"}
    json_data = json.dumps(data_dict)
    room = roomContext.room
    room.local_participant.send_text("hello",topic='chat')

# -----------------------------
# 3. Define nodes
# -----------------------------
def start_node(state: AgentState) -> AgentState:
    # Initialize with empty messages - system messages are handled by prompt template
    return {"messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"NARRATION_PLAN:\n{get_narration_plan()}")
    ]}


def call_llm(state: AgentState) -> AgentState:
    # Check if user wants to exit
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_input = last_message.content.lower().strip()
            if user_input in ["exit", "quit", "bye", "goodbye", "end", "stop"]:
                # Return a goodbye message and signal to exit
                goodbye_message = "Thank you for the knowledge transfer session!  Goodbye!"
                return {"messages": [AIMessage(content=goodbye_message)]}
    
    # Get initial response with tools
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(response)
    
    return state

def execute_tool(state: AgentState) -> AgentState:
    # Get the last message which should contain tool calls
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_id = tool_call["id"]
        tool = tools_by_name[tool_name]

        print(f"Tool input: {tool_input}")
        tool_response = tool.invoke(tool_input)
        print(f"Tool response: {tool_response}")

        # Add tool response to messages
        state["messages"].append(ToolMessage(
            content=json.dumps(tool_response),
            tool_call_id=tool_id,
        ))
    
    return state

def generate_final_response(state: AgentState) -> AgentState:
    # Get structured response
    structured_llm = llm.with_structured_output(Slide)
    final_response = structured_llm.invoke(state["messages"])
    
    # Return the AI response
    # send_data_to_room(final_response.slide_number)
    print(f"final_response slide_number : {final_response.slide_number}")
    return {"messages": [AIMessage(content=final_response.text)]}

def should_use_tools(state: AgentState) -> str:
    """Conditional function to determine if tools should be used"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "final"



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# StateGraph with conditional edges for tool handling
def create_graph() -> StateGraph:

    workflow = StateGraph(AgentState)

    workflow.add_node("start", start_node)
    workflow.add_node("llm_node", call_llm)
    workflow.add_node("tool_node", execute_tool)
    workflow.add_node("final_node", generate_final_response)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "llm_node")
    
    # Add conditional edge from llm_node
    workflow.add_conditional_edges(
        "llm_node",
        should_use_tools,
        {
            "tools": "tool_node",
            "final": "final_node"
        }
    )
    
    # After tool execution, go to final response
    workflow.add_edge("tool_node", "final_node")
    
    return workflow.compile()


async def entrypoint(ctx: JobContext):
    global roomContext
    roomContext = ctx
    graph = create_graph()

    agent = Agent(
        instructions="",
        llm=langchain.LLMAdapter(graph),
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        # any combination of STT, LLM, TTS, or realtime API can be used
        stt=openai.STT(model="gpt-4o-mini-transcribe", language="en"),
        tts=openai.TTS(model="gpt-4o-mini-tts",instructions="DO NOT SAY THE SLIDE NUMBER IN YOUR RESPONSE. Speak in a cheerful, positive, and professional tone. Talk in a medium pace", voice="alloy"),
        # tts = cartesia.TTS(model="sonic-2",voice="b911fa30-f9f2-45e0-8918-8671d24e61c8",api_key="sk_car_X1ffUzkFCuZRCtCKcXTDmH")
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
    await session.generate_reply()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
