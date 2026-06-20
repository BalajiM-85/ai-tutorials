# ============================================================
# 📦 INSTALLATION
# ============================================================
!pip install langchain langchain_experimental langchain_community --quiet
!pip install langchainhub pypdf langchain-groq ddgs --quiet

# ============================================================
# 📚 IMPORTS
# ============================================================
import os
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# ============================================================
# 🔑 API KEY
# ============================================================
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

# ============================================================
# 🤖 MODEL
# ============================================================
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# ============================================================
# 🛠️ TOOLS
# ============================================================
tools = [DuckDuckGoSearchRun()]

# ============================================================
# 🕵️ AGENT
# ============================================================
agent_executor = create_agent(model, tools)

# ============================================================
# 🚀 RUN
# ============================================================
query = "What are the top AI trends in 2026?"

response = agent_executor.invoke({
    "messages": [HumanMessage(content=query)]
})

# ✅ Extract last non-empty response
for msg in reversed(response["messages"]):
    if hasattr(msg, "content") and msg.content.strip():
        print(msg.content)
        break