from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

import os
from dotenv import load_dotenv

load_dotenv()
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

app = Flask(__name__)

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_input = event.message.text # event.message.text 是使用者傳來的訊息
    user_id = event.source.user_id
    print("使用者: ", user_input)    # user_input 是使用者傳來的訊息
    reply_message = generate_reply(user_input).content
    print("機器人: ", reply_message) # reply_message 是機器人回覆的訊息
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)],
            )
        )
from langchain_core.prompts import ChatPromptTemplate
from configs.setting import llm
def generate_reply(user_input):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一個台灣模擬的以房養老QA機器人，只用繁體中文與台灣用詞回答問題。",
            ),
            ("human", "{input}"),
        ]
    )

    LLMChain = prompt | llm
    ans = LLMChain.invoke(
        {
            "input": user_input,
        }
    )
    return ans

if __name__ == "__main__":
    app.run()



