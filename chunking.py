from sentence_transformers import SentenceTransformer
import faiss
model=SentenceTransformer("all-MiniLM-L6-V2")
def fileOpen():
    with open("data/os.txt","r",encoding="utf-8") as f:
        text=f.read()
    return text
def chunking(text):
    chunks=[]
    current=""
    for line in text.split("\n"):
        line=line.strip()
        if not line:
            continue
        if len(current)+len(line) <=500:
            current+=line+" "
        else:
            chunks.append(current.strip())
            current=line+" "
    if current:
        chunks.append(current.strip())
    return chunks
def embeddings(chunks):
    vector=model.encode(chunks)
    return chunks
text=fileOpen()
chunks=[]
chunks=chunking(text)
print(embeddings(chunks))