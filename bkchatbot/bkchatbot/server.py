import os
import sys
import logging
import tempfile
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
)
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 環境變數設定
logging.basicConfig(level=logging.INFO)
# 加載 .env 檔案
load_dotenv()

# 使用 os.getenv 讀取環境變數
api_key = os.getenv('OPENAI_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

# API客戶端初始化
# 初始化 ChatOpenAI
client = ChatOpenAI(
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
# 初始化Flask應用
app = Flask(__name__)

# 接收LINE平台傳來的各種事件（如用戶訊息或行動），並進行相應的處理
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# LINE API 配置
if channel_secret is None or channel_access_token is None:
    print('Please specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)
handler = WebhookHandler(channel_secret)
configuration = Configuration(access_token=channel_access_token)

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

# 生成回覆
def generate_reply(user_input, user_id):
    openai_result = openai(user_input, user_id)
    if hasattr(openai_result, 'content'):
        return openai_result.content
    return openai_result  # 假設沒有 content 屬性，直接回傳結果

# 處理文字消息事件
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
        reply_message = generate_reply(user_input, user_id)

        # 3️⃣ **回覆使用者訊息**
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)]
            )
        )

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
            reply_message = generate_reply(transcript, event.source.user_id)
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

# 定義 openai 函數，用於處理用戶消息
def openai(user_message, session_id):
    # 定義樂活養老計畫相關的關鍵字
    keyword_list = [
        "逆向", "抵押", "養老", "以房", "政策", "高齡", "不動產", "穩定", "安養", "生活",
        "現金", "流量", "到期", "清算", "分期", "還款", "貨款", "用途", "額度", "利率",
        "固定", "機動", "期限", "月領", "資金", "規定", "利息", "擔保品", "擔保", "申請",
        "所得稅", "增值", "解約", "條件", "分行", "申辦", "文件", "保證", "手續", "調查",
        "法院", "拍賣", "流動", "延期", "核貨", "定價", "身份證", "價值", "貸款", "估價",
        "年齡", "限制", "身份", "印章", "登記", "結束", "處理", "續貨", "住院", "治療",
        "中止", "年滿", "月付", "領取", "金額", "記錄", "續期", "重估", "保人", "詢問",
        "證實", "追價", "變動", "房地", "房產", "遺產", "終止", "契約", "每月", "年金",
        "房貸", "房屋", "房價", "服務", "相關", "條款", "樂活", "銀行", "土地", "專案",
        "計畫", "內容", "開始", "使用", "確保", "採用", "簡介", "房子", "辦理", "所得",
        "罰金", "賠款", "違約", "賠付", "賠償", "賠償金", "款項", "範例", "案例", "舉例",
        "示範", "示例", "特別", "續貸", "續借", "生病", "醫院", "醫療", "保險", "死亡",
        "繼承", "身故", "老年", "借貸", "資產", "財務狀況", "理財", "信用", "信用評分",
        "融資", "金融機構", "放款", "收入證明", "屋齡", "間房", "老房", "新房", "死掉",
        "指定", "繼承人", "繼承權", "財產", "權益", "繼續領", "領錢", "受益", "法定",
        "波動", "綁定", "綁約", "倒了", "倒閉", "倒賬", "倒賠", "會倒", "方案", "規劃",
        "股票", "投資", "基金", "ETF", "債券", "選擇權", "外匯", "匯率", "黃金", "白銀",
        "原油", "商品", "期指", "期貨", "期權", "租金", "租賃", "租屋", "租房", "租約",
        "租客", "租人", "天數", "月數", "幾天", "幾月", "費用", "費率", "了解", "瞭解",
        "知道", "懂得", "清楚", "明白", "模糊", "不清楚", "不懂", "取得", "資料", "資訊",
        "消息", "訊息", "通知", "通告", "公示", "公開", "獲取", "獲得", "得知", "得悉",
        "更好", "更棒", "更佳", "更優", "更讚", "更爛", "更差", "不好", "不棒", "不佳",
        "不優", "幫我", "解釋", "說明", "告訴", "詳細", "欠款", "負債", "負擔", "繳稅",
        "報稅", "稅金", "稅款", "稅務", "出售", "買賣", "交易", "買進", "賣出", "買入",
        "銷售", "購買", "購進", "購入", "專業", "專家", "諮詢", "洽詢", "請教", "請益",
        "請問", "請示", "依據", "根據", "據說", "最長", "最短", "最高", "最低", "最大",
        "最小", "最多", "最少", "最好", "最壞", "最優", "最爛", "最佳", "多少", "多久",
        "多長", "多高", "多低", "多大", "多小", "多好", "多壞", "多優", "多爛", "追討",
        "追償", "追回", "追究", "追查", "追蹤", "追踪", "幾歲", "幾年", "準備", "影響",
        "領的", "續約", "注意", "Q27", "Q5","資格", "條件", "規定", "限制", "限定","本金",
        "還本","老死",
    ]


    # 用戶輸入與樂活養老計畫相關，繼續處理
    input_question = user_message
    query_vector = embeddings.embed_query(input_question)
    search_results = qdrant_client.search(
        # collection_name="0418-PDF-fianl-data",
        collection_name="250206-PDF-3-small-final",
        query_vector=query_vector,
        limit=1
    )

    # 構建檢索結果的描述
    search_results_description = " ".join([
        f"找到的相關文件: {doc.payload['page_content']}" 
        for doc in search_results]) if search_results else "未找到與查詢相關的資訊。"

    print("向量資料搜尋結果:" + search_results_description)

    # 二次確認檢索結果的相關性
    if not any(keyword in search_results_description for keyword in keyword_list):
        return "找到的資訊並非關於樂活養老計畫的具體信息。"
        
    prompt_text = build_prompt_text(search_results_description)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{input}")
    ])
    runnable = prompt | client

    return runnable.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}}
    )
    # else:
    #     return "您好！我是土地銀行的樂活養老計畫AI助理。如果您有任何相關問題，需要服務請告訴我！"

# 定義 build_prompt_text 函數，用於構建提示文本
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
    return f"""
    {step_list}
    {search_results_description}
    """

# 啟動 Flask 應用
if __name__ == "__main__":
    app.run()
