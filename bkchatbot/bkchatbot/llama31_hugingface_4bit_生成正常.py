import os
import re
import sys
import logging
import tempfile
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, AudioMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
)
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from openai import OpenAI

# 載入環境變數
logging.basicConfig(level=logging.INFO)
load_dotenv()

# 讀取環境變數
api_key = os.getenv('OPENAI_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

# 初始化 Whisper 模型（用於語音轉文字）
whisper_client = OpenAI(api_key=api_key)

# 初始化 OpenAIEmbeddings（用於向量查詢）
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small",
)

# 初始化 QdrantClient
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# 設定 LLaMA 3.1 Instruct 模型
model_name = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 初始化 Flask 應用
app = Flask(__name__)

if channel_secret is None or channel_access_token is None:
    print('Please specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)
configuration = Configuration(access_token=channel_access_token)

@app.route("/callback", methods=['POST'])
def callback():
    """LINE 訊息回應端點"""
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    """處理 LINE 傳來的文字訊息"""
    user_input = event.message.text
    user_id = event.source.user_id

    reply_message = generate_rag_response(user_input, user_id)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)]
            )
        )

def generate_rag_response(user_input, user_id):
    """使用 RAG + LLaMA 3.1 產生回應"""
    return query_rag_llama3(user_input, user_id)

def query_rag_llama3(user_message, session_id):
    """執行向量檢索 + LLaMA 3.1 產生回應"""
    
    # 1️⃣ Qdrant 向量檢索
    query_vector = embeddings.embed_query(user_message)
    search_results = qdrant_client.search(
        collection_name="250206-PDF-3-small-final",
        query_vector=query_vector,
        limit=1
    )

    search_results_description = "\n".join([
        f"找到的相關文件: {doc.payload['page_content']}"
        for doc in search_results]) if search_results else "未找到與查詢相關的資訊。"

    print(f"🔍 向量資料搜尋結果: {search_results_description}")

    # 2️⃣ **手動構建符合 LLaMA 3.1 Instruct 的 Prompt**
    prompt_text = f"""你是一位專業的 AI 助理，專門解答土地銀行樂活養老房屋專案的問題。請根據以下資訊提供準確的回應，並使用條列式回答，回應不超過 100 字。

根據以下資訊回答問題：
{search_results_description}

使用者問題：{user_message}

回答：
"""

    # 3️⃣ 使用 LLaMA 3.1 生成回應
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            repetition_penalty=1.1,
            do_sample=False  # 確保生成一致性
        )

    final_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 4️⃣ **移除 LLaMA 可能回應的 Prompt 內容**
    final_reply = final_reply.replace(prompt_text, "").strip()

    return final_reply




if __name__ == "__main__":
    app.run()
