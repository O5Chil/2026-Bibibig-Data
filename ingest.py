from pathlib import Path
import shutil
import math

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


DATA_DIR = "data"
DB_DIR = "db"
BATCH_SIZE = 100


print("기존 벡터 DB 삭제 중...")

db_path = Path(DB_DIR)
if db_path.exists():
    shutil.rmtree(db_path)

print("PDF 읽는 중...")

docs = []

pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

if not pdf_files:
    print("data 폴더 안에 PDF가 없습니다.")
    exit()

for pdf in pdf_files:
    print(f"읽는 중 : {pdf.name}")
    loader = PyPDFLoader(str(pdf))
    docs.extend(loader.load())

print(f"\n총 {len(docs)}페이지를 읽었습니다.")


print("\n문단 분할 중...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"총 {len(chunks)}개의 문단으로 분할했습니다.")


print("\nEmbedding 모델 불러오는 중...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


print("\n벡터 DB 생성 시작...")

vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

total_batches = math.ceil(len(chunks) / BATCH_SIZE)

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    batch_number = i // BATCH_SIZE + 1

    print(
        f"[{batch_number}/{total_batches}] "
        f"{i + 1} ~ {i + len(batch)}번째 문단 저장 중..."
    )

    vector_db.add_documents(batch)

    print(
        f"[{batch_number}/{total_batches}] "
        f"{i + 1} ~ {i + len(batch)}번째 문단 저장 완료"
    )


print("\n====================================")
print("벡터 데이터베이스 생성 완료!")
print(f"총 {len(chunks)}개의 문단을 저장했습니다.")
print("저장 위치 : ./db")
print("====================================")