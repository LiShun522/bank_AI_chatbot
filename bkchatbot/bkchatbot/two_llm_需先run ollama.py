import os
import re
import sys
import logging
import tempfile
import threading
import time
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, AudioMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama  # 使用 Ollama 開源 LLM
from langchain.schema import HumanMessage

# 載入環境變數
logging.basicConfig(level=logging.INFO)
load_dotenv()

# 讀取環境變數
api_key = os.getenv('OPENAI_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

# 初始化向量查詢 Embeddings
embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# 初始化 QdrantClient
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# 初始化兩個不同的 ChatOllama 模型
llm1 = ChatOllama(model="deepseek-r1:8b", temperature=0.7, num_predict=500)
llm2 = ChatOllama(model="llama3.1", temperature=0.7, num_predict=500)

# 初始化 Flask 應用
app = Flask(__name__)

if channel_secret is None or channel_access_token is None:
    print('請設定 LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 環境變數。')
    sys.exit(1)

handler = WebhookHandler(channel_secret)
configuration = Configuration(access_token=channel_access_token)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 生成回覆（同時執行兩個 LLM）
def generate_reply(user_input, user_id):
    results = {"llm1": None, "llm2": None, "time1": None, "time2": None}

    def call_llm1():
        results["llm1"], results["time1"] = openai(user_input, user_id, llm1)

    def call_llm2():
        results["llm2"], results["time2"] = openai(user_input, user_id, llm2)

    # 使用 Threading 並行呼叫兩個 LLM
    thread1 = threading.Thread(target=call_llm1)
    thread2 = threading.Thread(target=call_llm2)
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()

    # 格式化回覆，包含模型的生成時間
    reply_message = f"""\
🔹 **版本1（DeepSeek）**
回覆時間: {results['time1']:.2f} 秒
{results['llm1']}

🔹 **版本2（Llama3.1）**
回覆時間: {results['time2']:.2f} 秒
{results['llm2']}
    """
    return reply_message

@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    user_input = event.message.text
    user_id = event.source.user_id

    reply_message = generate_reply(user_input, user_id)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)]
            )
        )

# LLM 呼叫函數（加入執行時間測量）
def openai(user_message, session_id, llm_model):
    start_time = time.time()  # 記錄開始時間

    input_question = user_message
    query_vector = embeddings.embed_query(input_question)

    search_results = qdrant_client.search(
        collection_name="250206-PDF-3-small-final",
        query_vector=query_vector,
        limit=1
    )
    
    search_results_description = " ".join([
        f"找到的相關文件: {doc.payload['page_content']}" 
        for doc in search_results
    ]) if search_results else "未找到與查詢相關的資訊。"
    
    print(f"向量資料搜尋結果（{llm_model.model}）:" + search_results_description)

    prompt_text = build_prompt_text(search_results_description)
    full_prompt = f"{prompt_text}\n使用者問題: {user_message}\n請依照上述指引，回答："

    reply_obj = llm_model.invoke([HumanMessage(content=full_prompt)])
    
    if hasattr(reply_obj, 'content'):
        raw_reply = reply_obj.content
    else:
        raw_reply = reply_obj

    # 移除 <think> 標籤內容
    final_reply = re.sub(r'<think>.*?</think>', '', raw_reply, flags=re.DOTALL).strip()
    
    end_time = time.time()  # 記錄結束時間
    execution_time = end_time - start_time  # 計算執行時間（秒）

    return final_reply, execution_time  # 回傳回覆與執行時間


# 構建提示文本
def build_prompt_text(search_results_description):
    step_list = """
    0. 確保回答的內容準確無誤，條列分段式回答。
    1. 作為臺灣土地銀行樂活養老房屋專案的專家，只回答相關問題。
    2. 不處理其他金融機構或非樂活養老相關的金融、保險或投資問題。
    3. 所有回答均使用臺灣正體中文，並且僅提供土地銀行的相關服務資訊。
    4. 若問題涉及不清楚或不相關內容，將引導使用者聯繫土地銀行客服。
    5. 範例與舉例，必須非常準確，不要有多餘的變化，依照資料本身回覆。
    6. 提供的舉例或範例限制為一個，如需更多信息請咨詢銀行。
    7. 對於提到其他銀行的問題，如國泰世華、台新、富邦等，一律不予回答。
    8. 只針對關鍵條款提供重點提示，不全面列出，具體細節建議諮詢銀行。
    9. 如果不清楚問題的細節，不準隨意回答，重新確認使用者意圖。
    10.請再次確認問題的具體內容，以提供最準確的信息。
    11.用戶問題不明確時，可以提示用戶樂活養老計畫的相關問題。
    12.簡潔回答，每次不超過100字，確保用詞淺顯易懂。
    13.所有回答均使用臺灣正體中文。
    14.領錢、領多少、有多少錢...等問題，只用搜尋範例給予提示。
    15.無保險理賠服務，確保回答準確。
    16.對於投資、理財、基金、股票、債券、ETF...等介紹諮詢皆不在服務範圍。
    17.63歲(含)即可申辦樂活養老。
    18.每次生成回覆前檢查內容是否遵照指南。
    """
    return f"{step_list}\n{search_results_description}"

if __name__ == "__main__":
    app.run()
