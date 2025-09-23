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

If it is not covered in the plan → respond: "I'll get back on this." and then continue narration from where you left off.

Completion

Once all sections are narrated, state: "That concludes the KT. I'm ready for any questions."

Important

Stay professional, clear, and structured.

Do not break character as the KT Narration Agent.

IMPORTANT: If the user says "exit", "quit", "bye", "goodbye", "end", or "stop", respond with a polite goodbye message and the session will end."""
NARRATION_PLAN = """
Narration Plan 

(Slide 1: Title Slide - ABC Pvt. Ltd. HR Policy Overview – Welcome New Employees!)

(0:00-0:30) Introduction & Welcome

Narrator: Hello and welcome to ABC Pvt. Ltd.! We're thrilled to have you join our team. This session is all about our HR policies, a vital part of your onboarding and your journey here at ABC. Think of our HR policies as a roadmap – they guide us on how we work together, treat each other, and build a successful career here. This overview will provide you with a solid understanding of key policies, ensuring a smooth and rewarding experience with us. We'll cover everything from your employment terms and benefits to our expectations around workplace conduct and performance management. So, let's get started!

(Slide 2: Agenda - What We'll Cover Today)

(0:30-1:00) Agenda Overview

Narrator: Today, we'll be covering six main areas. First, we'll look at the purpose and importance of HR policies at ABC. Then, we'll delve into our employment practices, including hiring and onboarding. Next up, we'll discuss compensation and benefits, followed by workplace conduct expectations. We'll then move onto performance management and career growth opportunities. Finally, we'll cover grievance redressal, disciplinary actions, and the procedures for employee separation. By the end of this session, you'll have a comprehensive understanding of the HR framework that governs our workplace.

(Slide 3: Why HR Policies Matter – Fairness, Compliance, Wellbeing)

(1:00-1:45) Purpose of HR Policies

Narrator: So, why do we have HR policies in the first place? Well, they serve several crucial purposes. Firstly, they ensure fairness for everyone. Policies create a level playing field, ensuring consistent treatment and equal opportunities for all employees. Secondly, they ensure compliance with all applicable laws and regulations, such as Indian labor laws. This protects both you and the company. And most importantly, HR policies promote employee wellbeing. They aim to create a safe, respectful, and supportive work environment where you can thrive. Our policies are designed to cover the entire employee lifecycle, from the moment you join us to the day you decide to move on.

(Slide 4: Employment Practices – Hiring, Onboarding, Contracts)

(1:45-2:45) Employment Practices

Narrator: Let's begin with Employment Practices. This covers the initial stages of your journey with ABC. Our Hiring and Onboarding process is structured to ensure we find the best talent and set you up for success. We have a robust recruitment process with background checks and are committed to equal opportunity employment. All new hires, like yourselves, complete a digital onboarding module and a manager-led induction program. This ensures you have all the information and support you need to get started. Following onboarding, you'll typically undergo a Probation Period of 6 months. This allows both you and ABC to assess whether the role is a good fit. Confirmation of your employment is based on performance and feedback from your manager. We also have various Contract Types – full-time, fixed-term, consultants, and interns. Each contract type comes with defined entitlements and obligations, which will be clearly outlined in your offer letter.

(Slide 5: Compensation & Benefits – Salary, Leave Structure, Other Benefits)

(2:45-4:30) Compensation & Benefits

Narrator: Now, let's talk about Compensation & Benefits. Your Salary and Incentives are determined using a benchmark-based pay structure, which is revised annually. This includes your fixed salary, potential performance bonus, and statutory benefits like Provident Fund (PF), Gratuity, and Employee State Insurance (ESI). These are mandatory benefits ensuring financial security. Moving on to our Leave Structure, we offer a range of leaves to cater to your needs. You are entitled to 12 days of Casual Leave (CL) per year for any personal needs. For sick days, you have 10 days of Sick Leave (SL) annually. Please note that for sick leaves exceeding 3 consecutive days, a medical certificate is required. You also accrue Earned Leave (EL) at a rate of 18 days per year. This can be carried forward as per company policy. For new parents, we offer generous Maternity Leave of 26 weeks, as mandated by the Maternity Benefit Act, and 10 days of Paternity Leave. Beyond these, ABC provides a comprehensive suite of Other Benefits, including health insurance for you and your dependents, a wellness allowance to support your health goals, and learning reimbursements to help you upskill and grow your career.

(Slide 6: Leave Structure - Quick Overview Table)

(4:30-4:45) Leave Structure Summary

Narrator: This slide provides a quick visual overview of our leave structure, summarizing the key details we just discussed. Please refer to the employee handbook for complete details and eligibility criteria.

(Slide 7: Workplace Conduct – Code of Conduct, Diversity & Inclusion, Confidentiality, Social Media)

(4:45-6:00) Workplace Conduct

Narrator: Let's move on to Workplace Conduct. This is crucial for creating a positive and productive work environment for everyone. We expect all employees to adhere to our Code of Conduct, which emphasizes professionalism, respect, and integrity in all interactions. Our Diversity & Inclusion policy reflects our commitment to creating a workplace where everyone feels valued and respected. We have a zero-tolerance policy towards harassment or discrimination of any kind. Confidentiality is paramount at ABC. You will be handling sensitive information, including intellectual property, client data, and internal company information. It's crucial to protect this information and maintain confidentiality at all times. Finally, our Social Media Policy outlines guidelines for responsible online behavior. As a representative of ABC, you should not share any confidential or sensitive company information online.

(Slide 8: Work Hours & Flexibility – Standard Hours, Flexible Work, Overtime)

(6:00-7:00) Work Hours & Flexibility

Narrator: Let's discuss Work Hours and Flexibility. Our Standard Hours are 9 AM to 6 PM, Monday to Friday, amounting to a 40-hour work week. We understand the importance of work-life balance and offer Flexible Work options. Hybrid and Work-From-Home (WFH) arrangements are allowed for eligible roles, subject to manager approval. This allows you to manage your schedule and work environment to a certain extent, while ensuring you meet your professional commitments. In situations where Overtime is required, it will be compensated either with payment, as per statutory norms, or with compensatory time off. Please discuss overtime arrangements with your manager.

(Slide 9: Performance Management – Appraisal Cycle, Performance Reviews, Career Growth, L&D)

(7:00-8:30) Performance Management

Narrator: Now, let's talk about Performance Management. At ABC, we believe in continuous growth and development. Our Appraisal Cycle is conducted twice a year – at mid-year and annually. These reviews are designed to provide you with regular feedback and help you achieve your goals. Our Performance Reviews are a comprehensive process, incorporating self-evaluation, manager review, and peer or 360° feedback, depending on your role. This holistic approach ensures a fair and accurate assessment of your performance. We are committed to providing Career Growth opportunities to our employees. Promotions are linked to sustained performance and the demonstration of skills growth. We offer a range of L&D Initiatives to support your professional development. These include training programs, access to online learning portals, and mentorship programs. We encourage you to take advantage of these resources to enhance your skills and advance your career here at ABC.

(Slide 10: Grievance & Discipline – Grievance Redressal, Disciplinary Process, POSH Policy)

(8:30-10:00) Grievance & Discipline

Narrator: Let's move onto Grievance & Discipline. We strive to create a workplace where concerns can be addressed fairly and effectively. Our Grievance Redressal process provides channels for you to raise any concerns you may have. You can escalate your grievances through your HR Business Partner or utilize our anonymous whistleblower hotline. We take all grievances seriously and are committed to resolving them promptly. In cases of misconduct or policy violations, we follow a clear Disciplinary Process. This is a progressive action model, starting with a warning, followed by an HR review, and potentially leading to termination if necessary. The severity of the disciplinary action depends on the nature of the violation. We also have a robust POSH Policy in place, which addresses sexual harassment in the workplace. An Internal Complaints Committee (ICC) is responsible for handling sexual harassment cases, ensuring compliance with all statutory requirements. We are committed to creating a safe and respectful work environment for everyone.

(Slide 11: Separation & Exit – Resignation, Exit Process, Final Settlement)

(10:00-11:00) Separation & Exit

Narrator: Now, let's discuss Separation & Exit. We hope you have a long and successful career with ABC. However, should you decide to leave, we have a clear process in place. Our Resignation Policy requires a 30-day notice period for voluntary exit. This allows us to plan for your transition and ensure a smooth handover of your responsibilities. The Exit Process includes a clearance process, where you'll return company property and complete necessary paperwork. It also includes knowledge transfer, ensuring your work is properly documented and can be taken over by your colleagues. Finally, you'll have an exit interview to provide feedback on your experience at ABC. Your Final Settlement, covering salary dues, leave encashment, and statutory payments, will be completed within 30 days of your last working day.

(Slide 12: Compliance & Policy Updates – Annual Review, Communication & Acknowledgment)

(11:00-11:30) Compliance & Policy Updates

Narrator: It's important to note that our HR policies are dynamic and subject to change. The HR policies are reviewed annually by the HR leadership team to ensure they are up-to-date and aligned with best practices and legal requirements. Any Updates are communicated through announcements on our Confluence space and require mandatory employee acknowledgment. We encourage you to stay informed about any policy changes to ensure compliance and a smooth working experience.

(Slide 13: Learning Objectives Revisited - Understand, Familiarize, Learn, Recognize, Understand)

(11:30-12:00) Conclusion & Key Takeaways

Narrator: To recap, in this session, we covered a lot of ground. We aimed to help you understand the purpose and importance of HR policies at ABC Pvt. Ltd. We walked you through key employment practices, benefits, and leave policies. You learned about our expectations for workplace conduct, including our code of conduct and confidentiality obligations. We highlighted the opportunities for performance management and career growth and clarified the procedures for grievances, disciplinary actions, and employee exits. Remember, these policies are in place to support you and create a positive work environment. You can always find the full details in the Employee Handbook on our HR Confluence Space. If you have any questions, please don't hesitate to reach out to your manager or HR Business Partner. Welcome aboard, and we wish you a successful journey with ABC Pvt. Ltd.!

(Slide 14: Thank you & Contact Information - HR Department Email/Phone)

(12:00) End

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

# -----------------------------
# 4. Conditional routing
# -----------------------------
def should_continue(state: AgentState) -> str:
    # Check the last message for ending conditions
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and "bye" in last_message.content.lower():
            return "end"
    return "continue"


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
