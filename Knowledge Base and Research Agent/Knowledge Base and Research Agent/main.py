import chainlit as cl
from agents import Agent, Runner, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from model_config import model_config
from tools import web_search
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import fitz  # PyMuPDF

# Load .env
load_dotenv()

# -------------------------
# Session & Config
# -------------------------
session = SQLiteSession("rag_session", "conversations.db")
config = model_config()
client = OpenAI()

# -------------------------
# Load PDF & Embeddings
# -------------------------
qa_pairs = []   # list of (question, answer, embedding)

def load_pdf(path="questions_100.pdf"):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    # Assume Q&A format: "Q: ... A: ..."
    chunks = text.split("Q:")
    for c in chunks[1:]:
        if "A:" in c:
            q, a = c.split("A:", 1)
            q = q.strip()
            a = a.strip()
            emb = embed_text(q)
            qa_pairs.append((q, a, emb))

def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def find_best_answer(user_query: str, threshold=0.4):
    if not qa_pairs:
        return None
    query_emb = embed_text(user_query)
    sims = [([query_emb], [qa[2]])[0][0] for qa in qa_pairs]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim >= threshold:
        return qa_pairs[best_idx][1]
    return None

# Load PDF at startup
load_pdf()

# -------------------------
# Agents
# -------------------------
summarize_agent = Agent(
    name="SummarizeAgent",
    instructions="""
You are a summarizer agent.
- Given a Q&A pair from the PDF, return the answer in a user-friendly way.
- If user prefers Urdu, respond in Urdu.
- Be concise and clear.
"""
)

main_agent = Agent(
    name="MainAgent",
    instructions="""
You are the main chat assistant.
- Handle greetings, casual talk, and small talk.
- If user asks a factual question, system will try RAG from PDF first, then web search.
- Keep tone friendly and emoji-friendly 😊.
""",
    tools=[web_search]
)

# -------------------------
# Starters
# -------------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="❓ Ask Question", message="What is Artificial Intelligence?"),
        cl.Starter(label="💬 General Chat", message="Hello! How are you?"),
        cl.Starter(label="🆘 Help", message="Can you guide me about this project?")
    ]

# -------------------------
# Chat Handler
# -------------------------
@cl.on_message
async def handle_message(message: cl.Message):
    user_text = message.content.strip()
    thinking = cl.Message(content="🤔 Processing...⏳")
    await thinking.send()

    # Step 1: Try PDF similarity
    pdf_answer = find_best_answer(user_text)
    if pdf_answer:
        thinking.content = pdf_answer + " 😊"
        await thinking.update()
        return

    # Step 2: Fallback to main agent (LLM + web search)
    result = Runner.run_streamed(
        main_agent,
        input=user_text,
        run_config=config,
        session=session,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await thinking.stream_token(event.data.delta)
    thinking.content = result.final_output
    await thinking.update()