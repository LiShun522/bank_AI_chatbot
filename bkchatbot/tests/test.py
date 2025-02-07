import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# 初始化 ChatOpenAI
llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.25, 
    max_tokens=500, 
    model="gpt-4o",
)

user_input = input("請輸入問題: ")
prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一個台灣模擬的以房養老QA機器人，只用繁體中文與台灣用詞回答問題。"),
            ("human", "{input}"),
        ]
    )

LLMChain = prompt | llm
ans = LLMChain.invoke(
    {
        "input": user_input,
    }
)

print(ans.content)