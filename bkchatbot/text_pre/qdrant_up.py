import os
from dotenv import load_dotenv
# from pdf2image import convert_from_path
# import pytesseract
from qdrant_client import QdrantClient

from langchain.vectorstores.qdrant import Qdrant

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PDFPlumberLoader

# 載入.env文件中的變量
load_dotenv('.env')

# 從.env文件獲取API Key
api_key = os.getenv('OPENAI_API_KEY')

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
# 設定OpenAI的環境變數
os.environ["OPENAI_API_KEY"] = api_key

# 初始化嵌入模型，指定模型和API版本
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-ada-002"
# )
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 指定前兩個 PDF 文件的路徑
file_path1 = r"c:\Users\gn012\OneDrive\Desktop\ProJect\bank\土銀專案\PDF\臺灣土地銀行 - 個金單一服務平台.pdf"
file_path2 = r"c:\Users\gn012\OneDrive\Desktop\ProJect\bank\土銀專案\PDF\樂活養老貸款 範例.pdf"

# 創建兩個 PDFPlumberLoader 實例
loader1 = PDFPlumberLoader(file_path1) 
loader2 = PDFPlumberLoader(file_path2)





# OCR 的 PDF 文件路徑
# file_path3 = r"D:\bank\土銀專案\PDF\樂活養老懶人包DM.pdf"

# # 將 PDF 轉換為圖像
# pages = convert_from_path(file_path3, dpi=1000)

# # 對每一頁進行 OCR 處理
# for page_image in pages:
#     text = pytesseract.image_to_string(page_image, lang='eng+chi_tra')  # 指定語言如果非英文
#     data.append(text)

# from PIL import Image, ImageFilter, ImageOps
# # 圖像預處理和 OCR 處理
# for page_image in pages:
#     # 轉換為灰階
#     page_image = page_image.convert('L')
#     # 自動對比度
#     # page_image = ImageOps.autocontrast(page_image)
#     # # 中值濾波去噪
#     # page_image = page_image.filter(ImageFilter.MedianFilter())
#     # # 二值化
#     # page_image = page_image.point(lambda x: 0 if x < 140 else 255, '1')
    
#     # 使用 pytesseract 進行 OCR
#     text = pytesseract.image_to_string(page_image, lang='chi_tra')
#     data.append(text)

# 現在 data 列表包含了前兩個 PDF 文件的文本以及經過 OCR 處理的第三個文件的文本




# 讀取PDF檔案方法二
from langchain_community.document_loaders.pdf import PyPDFLoader
file_path3 = r"c:\Users\gn012\OneDrive\Desktop\ProJect\bank\土銀專案\PDF\樂活養老懶人包DM.pdf"
loader3 = PyPDFLoader(file_path3,extract_images=True)
loader3_img = PDFPlumberLoader(file_path3,extract_images=True)

# 加載前兩個 PDF 文件的數據
data1 = loader1.load()
data2 = loader2.load()
data3 = loader3.load()
data3_img = loader3_img.load()
# 我要將內容全部轉換成繁體中文

#我要將data3_img的每一行打印出來
for i in data3_img:
    print(i)
# 將前兩個文件的數據合併
data = data3
data = data1 + data2 + data3
print(data)
# 分割文本方法一(財務報表勝出)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # separator調整為空白優先，這樣我們的 overlap 才會正常運作
    separators=[" ", "\n"],
    chunk_size = 150,
    chunk_overlap  = 25,
    length_function = len,
    add_start_index = True,
)

# 分割文本方法二(保險條款勝出)
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter()


# 將切割後的文本存入all_splits
all_splits = text_splitter.split_documents(data)
#我要打印出all_splits的每一行
for i in all_splits:
    print(i)
# splitter = CharacterTextSplitter(
#     separator=" \n",
#     chunk_size=500, 
#     chunk_overlap=10, 
#     length_function=len,
# )
# parts = splitter.split_documents(data)
# for part in parts:
#     print(part)

#建立向量資料庫
# qdrant = QdrantClient("http://127.0.0.1:6333") 


qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)

#刪除舊的collection,重新上傳
qdrant_client.delete_collection("0414-PDF")

db = Qdrant.from_documents(
    all_splits,
    # parts,
    embeddings,
    # url="http://127.0.0.1:6333",
    url=qdrant_url, 
    api_key=qdrant_api_key, 
    collection_name="250206-PDF-3-small",
)

# 取得以向量資料庫的搜尋演算法為主的檢索器
retriever = db.as_retriever()