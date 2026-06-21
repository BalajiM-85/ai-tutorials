# ============================================================
# 📦 INSTALLATION
# ============================================================
!pip install langchain langchain_experimental langchain_community --quiet
!pip install langchainhub pypdf langchain-groq --quiet
!pip install langchain-huggingface --quiet
!pip install faiss-cpu --quiet

# ============================================================
# 📚 IMPORTS
# ============================================================
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from google.colab import userdata

# ============================================================
# 🔑 CONFIGURATION
# ============================================================
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

PDF_URL  = "https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Microsoft_Earnings/TranscriptQandAFY25q4.pdf"
PDF_PATH = "TranscriptQandAFY25q4.pdf"

CHUNK_SIZE    = 300
CHUNK_OVERLAP = 30
TOP_K         = 3
MODEL_NAME    = "openai/gpt-oss-120b"

# ============================================================
# 📄 STEP 1 — LOAD DOCUMENT
# ============================================================
!wget {PDF_URL} -O {PDF_PATH}

loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

full_text = "".join(page.page_content for page in pages)

print(f"Pages      : {len(pages)}")
print(f"Lines      : {len(full_text.splitlines())}")
print(f"Words      : {len(full_text.split())}")
print(f"Characters : {len(full_text)}")

# ============================================================
# ✂️  STEP 2 — CHUNK TEXT
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = text_splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

# ============================================================
# 🔍 STEP 3 — EMBED & INDEX (FAISS)
# ============================================================
embeddings  = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ============================================================
# 🤖 STEP 4 — INITIALISE LLM
# ============================================================
model = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
)

# ============================================================
# 🔗 STEP 5 — BUILD RAG CHAIN
# ============================================================
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# ============================================================
# 🚀 STEP 6 — QUERY
# ============================================================
query    = "What is the LinkedIn revenue?"
response = rag_chain.invoke(query)
print(response)

# ============================================================
# 🚀 STEP 7 — Create RAG Tool for Agentic RAG
# ============================================================
from langchain.agents import create_agent
from langchain_community.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class RAGInput(BaseModel):
    query: str

rag_tool = StructuredTool(
    name="MS_Earnings_RAG",
    func=lambda query: rag_chain.invoke(query),
    description="Useful for answering questions about MicroSoft Earnings",
    args_schema=RAGInput
)

tools=[rag_tool]

# ============================================================
# 🕵️ AGENT
# ============================================================
agent_executor = create_agent(model, tools)

# ============================================================
# 🚀 RUN
# ============================================================
query = "What is the news about copilot adoption?"

response = agent_executor.invoke({
    "messages": [HumanMessage(content=query)]
})

# ✅ Extract last non-empty response
for msg in reversed(response["messages"]):
    if hasattr(msg, "content") and msg.content.strip():
        print(msg.content)
        break