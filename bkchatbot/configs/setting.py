import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# 初始化 ChatOpenAI
llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.25, 
    max_tokens=500, 
    model="gpt-4o",
)
# 初始化 Whisper 模型
whisper_client = OpenAI(
    api_key=api_key
)
# 初始化 OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small",
)
# 初始化 QdrantClient
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)