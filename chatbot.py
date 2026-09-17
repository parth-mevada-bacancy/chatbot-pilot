"""
Final Mini AI System — Section 8 Self Exercise (Practical Generative AI Workshop)

Combines everything from the workshop into one runnable chatbot:
  Chat Interface -> Conversation History -> Retriever -> Context -> Gemini -> Response

Install deps first:
    uv add langchain langchain-core langchain-google-genai google-generativeai langchain-text-splitters

Run:
    uv run python chatbot.py
"""

import os
import logging
from getpass import getpass

# Silence the google-genai SDK's benign "use Chat.send_message instead" advisory
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai


# ---------------------------------------------------------------------------
# 1. API key (never hardcoded)
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def get_text(response):
    """Return plain text of an LLM response, whether `.content` is a string
    or a list of content blocks (e.g. text + an internal thought-signature block)."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)


# ---------------------------------------------------------------------------
# 3. Source document — loaded from context.txt at startup
# ---------------------------------------------------------------------------
CONTEXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context.txt")

with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
    sample_document = f.read()

docs = [Document(page_content=sample_document, metadata={"source": "context.txt"})]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 4. Embeddings + vector store + retriever
# ---------------------------------------------------------------------------
embedding_model_name = None
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        embedding_model_name = m.name
        break

embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ---------------------------------------------------------------------------
# 5. RAG + memory chatbot
# ---------------------------------------------------------------------------
conversation_history = []  # list of HumanMessage / AIMessage, growing each turn


def rag_chat(user_message):
    # 1. Retrieve document context relevant to THIS message
    relevant_docs = retriever.invoke(user_message)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # 2. Build fresh instructions (including retrieved context) for this turn
    system_message = SystemMessage(content=(
        "You are a helpful support assistant for Wavelength Music customers. "
        "Use the CONTEXT below if it's relevant to the question. "
        "If the answer isn't in the context AND isn't something already established "
        "earlier in this conversation, say you don't have that information. "
        "Do not make up plan details or policy details.\n\nCONTEXT:\n" + context
    ))

    # 3. Combine: system instructions + everything said so far + the new message
    messages = [system_message] + conversation_history + [HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    reply_text = get_text(response)

    # 4. Record this turn so future turns remember it
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=reply_text))

    return reply_text


# ---------------------------------------------------------------------------
# 6. Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Wavelength Music assistant ready. Type 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            break
        print("Wavy:", rag_chat(user_input))
