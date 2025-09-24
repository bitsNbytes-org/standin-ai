import logging
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

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

load_dotenv()

# pip install langchain langgraph openai

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph

# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# -----------------------------
# 2. Create LLM + prompt template
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt for the KT Narration Agent
SYSTEM_PROMPT = """You are an AI Knowledge Transfer (KT) Narration Agent.

Your role is to deliver knowledge transfer on behalf of an engineer to new engineers, based on a provided narration plan.

Follow these rules:

Conversation History Awareness

If the conversation history is empty, begin narration from the start of the narration plan.

If there is existing conversation history, review the last few items:

If narration is in progress, continue from where it left off.

If narration is already completed, respond: "The KT narration is completed. I'm ready for any questions."

Narration Style

Narrate step by step, following the narration plan.

After each section, pause briefly for possible interruptions.

If no one interrupts, continue narrating automatically.

Question Handling

If a question is asked during narration:

If the answer exists in the narration plan → provide it.

If it is not covered in the plan → respond: "I dont have clear answer for it now but I'll get back on this." and then continue narration from where you left off.

If the user asks the questions which is going to be covered next the narration plan then say we will cover it in the upcoming section.
For this check the conversation history if its discussed in the conversation history then provide the answer. if its not discussed then say will cover it in the upcoming section

Completion

Once all sections are narrated, state: "That concludes the KT. I'm ready for any questions."

Important

Stay professional, clear, and structured.

Do not break character as the KT Narration Agent.

IMPORTANT: If the user says "exit", "quit", "bye", "goodbye", "end", or "stop", respond with a polite goodbye message and the session will end."""
NARRATION_PLAN = """
Narration Plan 
# Python Programming: Key Concepts - A Quick Overview

Duration: 70.0 seconds
Slides: 3

==================================================

## Slide 1
Duration: 25.0s
Image: Not generated

Hey developers! Let's dive into a quick overview of Python's core concepts. We'll cover the essentials to get you up and running. First up: Python syntax and structure. Python is all about readability, right? It uses indentation to define code blocks – think clean, easy-to-follow code. No curly braces or semicolons needed! Just a simple new line to end a statement. This keeps things concise and helps you focus on the logic, not the syntax.

------------------------------

## Slide 2
Duration: 25.0s
Image: Not generated

Next, let's talk data types and variables. Python's got your back with common types like integers, floats, strings, and lists. The best part? Variables are dynamically typed. That means you don't have to declare a type upfront. You just assign a value, and Python figures it out. This flexibility speeds up your coding process and reduces those pesky type-related errors. It's all about efficiency!

------------------------------

## Slide 3
Duration: 20.0s
Image: Not generated

Finally, let's touch on functions and modules. Functions are defined with the 'def' keyword – super straightforward. They can take parameters and return values, just like you'd expect. And to keep your code organized, you can break it into modules – basically, Python files that you can import and reuse. This modular approach encourages code reuse and makes your projects much easier to manage, especially as they grow. That's it! We've covered the main areas. Happy coding!
------------------------------
"""

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
    response = llm.invoke(conversation)
    
    # Return the AI response
    return {"messages": [AIMessage(content=response.content)]}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# a simple StateGraph with a single GPT-4o node
def create_graph() -> StateGraph:
    openai_llm = init_chat_model(
        model="openai:gpt-4o-mini",
    )

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
        instructions="You are an AI agent providing knowledge transfer.",
        llm=langchain.LLMAdapter(graph),
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        # any combination of STT, LLM, TTS, or realtime API can be used
        stt=openai.STT(model="gpt-4o-mini-transcribe", language="en"),
        tts=openai.TTS(model="gpt-4o-mini-tts", 
                voice="alloy", 
                instructions="Speak in a cheerful, positive, and professional tone. Talk in a medium pace"),
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
    #await session.generate_reply(instructions="Welcome! I'm here to provide knowledge transfer about ABC Pvt. Ltd. HR policies. Let me start by giving you an overview of our HR framework. You can say 'exit' anytime to end this session.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
