import os
# from config.load_key import load_key
from langchain_core.documents import Document
from langchain_postgres import PGVector
#from langchain_postgres.vectorstores import PGVector

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://postgres:123456@localhost:5432/Couplet"  # Uses psycopg3!
collection_name = "couplet"
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
load_dotenv()

# EmbeddingKey=(os.environ["EmbeddingKey"])
# EmbeddingUrl=(os.environ["EmbeddingUrl"])
# EmbeddingModel=(os.environ["EmbeddingModel"])
# RerankModel=(os.environ["RerankModel"])
DASHSCOPE_API_KEY=(os.environ["DASHSCOPE_API_KEY"])
DashScopeEmbeddingModel=(os.environ["DashScopeEmbeddingModel"])

embedding_model=DashScopeEmbeddings(
    model=DashScopeEmbeddingModel
)
vector_store = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
lines=[]
#with open("./resource/couplet.csv","r",encoding="gbk") as file:
id=1
with open("MulitAgent/couplet.csv","r",encoding="gbk") as file:
    for line in file:
        id=id+1
        print(line)
        doc=Document(page_content=line,metadata={"id":id})
        lines.append(doc)

vector_store.add_documents(lines, ids=[doc.metadata["id"] for doc in lines])