import os
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
from linebot.v3.messaging.models.show_loading_animation_request import ShowLoadingAnimationRequest
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


# 使用 OpenAI Whisper 進行語音轉文字
def speech_to_text(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh"  # 指定語言為中文
            )
        transcript = transcription.text.strip()
        logging.info(f"Transcript: {transcript}")
        return transcript
    except Exception as e:
        logging.error(f"Error processing audio file with OpenAI Whisper: {e}")
        return None


# 處理音頻消息事件
@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        audio_data = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )

        with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        transcript = speech_to_text(temp_audio_path)
        if transcript:
            # 使用從語音識別得到的文字進行回覆生成
            reply_message = generate_rag_response(transcript, event.source.user_id)
        else:
            reply_message = "無法辨識語音內容"

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)],
            )
        )

        os.remove(temp_audio_path)


@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    """處理 LINE 傳來的文字訊息"""
    user_input = event.message.text
    user_id = event.source.user_id  # 取得使用者 ID（作為 chatId）

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # 1️⃣ **顯示思考動畫**
        try:
            show_loading_animation_request = ShowLoadingAnimationRequest(chatId=user_id)
            line_bot_api.show_loading_animation(show_loading_animation_request)
        except Exception as e:
            logging.error(f"顯示思考動畫失敗: {e}")

        # 2️⃣ **生成回應**
        reply_message = generate_rag_response(user_input, user_id)

        # 3️⃣ **回覆使用者訊息**
        line_bot_api.reply_message(
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

    # 2️⃣ **符合 LLaMA 3.1 Instruct 格式**
    prompt_text = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
你是一位專業的 AI 助理，專門解答土地銀行樂活養老房屋專案的問題。
請根據以下資訊提供準確的回應:
專家角色：僅回答臺灣土地銀行樂活養老房屋專案相關問題，不處理其他銀行或金融產品。
回應原則：回答須準確無誤，條列式、正體中文、簡潔不超過 100 字。
限制範圍：
不回答保險、投資、理財、基金、股票、ETF 等問題。
不回應其他銀行（如國泰世華、台新、富邦等）相關問題。
若問題不明確或超出範圍，引導聯繫土地銀行客服。
舉例與範例：提供的範例須精確，僅限 1 個，依照資料回應，不做變化。
資訊重點：
只提供關鍵條款摘要，完整細節建議諮詢銀行。
63 歲（含）可申辦樂活養老。
「領錢」、「領多少」等問題僅提供搜尋範例提示。
內容驗證：回答前確認問題細節，確保回應符合指南，不隨意推測。
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
根據以下資訊回答問題：
{search_results_description}

使用者問題：{user_message}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    # 3️⃣ **使用 LLaMA 3.1 生成回應**
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,   # 限制最多 150 個 token
            temperature=0.3,      # 降低隨機性，確保回答穩定
            repetition_penalty=1.1,
            do_sample=False       # 關閉隨機取樣，提高一致性
        )

    final_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    final_reply = final_reply.split("assistant", 1)[-1].strip(": \n")

    return final_reply

if __name__ == "__main__":
    app.run()
