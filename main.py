from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, tempfile, shutil
from dotenv import load_dotenv
import uvicorn

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent_executor = None

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    global agent_executor
    if not agent_executor:
        return {"error": "Agent not ready"}
    
    result = agent_executor.invoke({"input": request.message})
    return {"response": result["output"]}

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        docs = []
        for file in files:
            path = f"{temp_dir}/{file.filename}"
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1")
        global vectorstore
        vectorstore = FAISS.from_documents(splits, embeddings)
        return {"status": "success"}
    finally:
        shutil.rmtree(temp_dir)

@app.get("/status")
async def status():
    return {"ready": agent_executor is not None}

@app.on_event("startup")
async def startup():
    global agent_executor
    llm = ChatOpenAI(model="mistralai/mistral-7b-instruct:free", api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    tools = [TavilySearchResults(max_results=3)] if os.getenv("TAVILY_API_KEY") else []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Use tools to answer accurately."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

if __name__ == "__main__":
   
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

