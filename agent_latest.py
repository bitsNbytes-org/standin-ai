import logging
from typing import Annotated, TypedDict, AsyncIterable

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
import json
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    ModelSettings,
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
from pydantic import BaseModel

# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# -----------------------------
# 2. Create LLM + prompt template
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



def get_narration_plan():
    with open("output/presentation_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    narration_plan = []
    for slide in data.get("slides", []):
        narration_plan.append(
            f"(Slide {slide['slide_number']})\nNarrator: {slide['narration_text']}\n"
        )
    return "\n".join(narration_plan)

# System prompt for the KT Narration Agent
SYSTEM_PROMPT = """You are an AI Knowledge Transfer (KT) Agent helping onboard new engineers. Your job is to walk them through technical content based on a predefined narration plan, in a way that feels like a natural, human conversation.

Your Behavior Guidelines

1. Starting Off

If there’s no prior conversation, start from the beginning of the narration plan.

If there's an ongoing discussion, review the last few exchanges:

If you're mid-way through the KT, pick up from where you left off.

If the KT is already done, say:
"We've covered everything in the KT. Let me know if you have any questions."

2. Tone and Interaction Style

Speak clearly, professionally, and as if you’re talking directly to someone.

Avoid robotic or overly formal language.

Don’t say you're "narrating" or use phrases like "as a narration agent" — just talk like a knowledgeable colleague.

Present information step-by-step, in line with the narration plan.

After each section, pause and ask:
"Any questions so far?"

If there's no response or interruption after a short pause, smoothly continue to the next part.

3. Handling Questions

If someone asks something:

Covered in the plan? → Answer it clearly.

Not covered? → Say:
"I don’t have a clear answer for that right now, but I’ll follow up on it."
Then continue from where you left off.

Related to an upcoming section?

If already discussed in the conversation history → answer it.

If not discussed yet → say:
"That’s coming up shortly — I’ll cover it in a bit."

Also ask Do you have any other questions on that?"

Wait briefly for a response before continuing.

4. When the KT is Complete

Once the final section is done, say:
"That wraps up the KT. Feel free to ask any questions you might have."

5. Session End

If the user says "exit," "quit," "bye," "goodbye," "end," or "stop", respond politely and close the session.
Example:
"Thanks for your time. Take care!"

Important Notes:

Stay on-topic and aligned with the narration plan.

Don’t break character — you're a knowledgeable AI assisting with KT.

Maintain a helpful, clear, and supportive attitude throughout.
Output Format:

{
    "text": "text of the slide",
    "_sldx": "slide number"
}
"""

NARRATION_PLAN = f"{get_narration_plan()}"

# -----------------------------
# 3. Define nodes
# -----------------------------
def start_node(state: AgentState) -> AgentState:
    # Initialize with empty messages - system messages are handled by prompt template
    return {"messages": []}

class Slide(BaseModel):
    text: str
    _sldx: str

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
    structured_llm = llm.with_structured_output(Slide)
    response = structured_llm.invoke(conversation)
    response_text = response.text
    print("response_slide_number", response)
    # Return the AI response
    return {"messages": [AIMessage(content=response_text)]}


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

    # Create a custom agent class that uses our TTS node
    class CustomAgent(Agent):
        async def tts_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncIterable[rtc.AudioFrame]:
            """
            Custom TTS node with text and audio processing capabilities.
            This allows for custom text processing before TTS and custom audio processing after.
            """
            # Insert custom text processing here
            # You can modify the text stream before it goes to TTS
            # Slice the AsyncIterable to get first 5 characters
            
            async def slice_text_generator():
                
                async for text_chunk in text:
                    
                    # If the current chunk contains "sldx", skip yielding anything and reset chunks
                    if "_" in text_chunk:
                        break
                    else:
                        yield text_chunk
                        
            
            # Use the default TTS node for actual speech synthesis
            async for frame in Agent.default.tts_node(self, slice_text_generator(), model_settings):
                # Insert custom audio processing here
                # You can modify the audio frame before yielding
                # Example: Apply audio effects, volume adjustment, etc.
                yield frame

    agent = CustomAgent(
        instructions="You are an AI agent providing knowledge transfer. You are talking to nishanth, Wait for their answer then explain your topic",
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
    await session.say("Hey Hi Welcome to the session")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
