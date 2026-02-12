# ===============================
# 📦 IMPORTS
# ===============================
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ===============================
# 🔹 LOAD MODELS
# ===============================

print("🔄 Loading models...")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generator model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
print("✅ Models loaded.")


# ===============================
# 🔹 FILE READING
# ===============================
def file_open():
    with open("data/os.txt", "r", encoding="utf-8") as f:
        return f.read()


# ===============================
# 🔹 CHUNKING
# ===============================
def chunk_text(text, chunk_size=500):
    chunks = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if len(current) + len(line) <= chunk_size:
            current += line + " "
        else:
            chunks.append(current.strip())
            current = line + " "

    if current:
        chunks.append(current.strip())

    return chunks


# ===============================
# 🔹 CREATE EMBEDDINGS
# ===============================
def create_embeddings(chunks):
    vectors = embedding_model.encode(chunks)
    return np.array(vectors).astype("float32")


# ===============================
# 🔹 BUILD FAISS INDEX
# ===============================
def build_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


# ===============================
# 🔹 SEARCH FUNCTION
# ===============================
def search(query, index, chunks, k=3):
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results


# ===============================
# 🔹 GENERATE FINAL ANSWER (Improved Prompt)
# ===============================
def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an educational assistant.

Using ONLY the context provided below, write a detailed, clear, and well-structured answer.
If there are multiple points, list them clearly with explanations.
Do not add information that is not in the context.

Context:
{context}

Question:
{query}

Detailed Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = llm.generate(
        **inputs,
        max_length=400,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ===============================
# 🚀 MAIN PIPELINE
# ===============================
def main():
    print("📖 Loading document...")
    text = file_open()

    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"📦 Total Chunks Created: {len(chunks)}")

    print("🧠 Creating embeddings...")
    vectors = create_embeddings(chunks)

    print("🔍 Building FAISS index...")
    index = build_faiss_index(vectors)

    print("\n✅ RAG system ready!\n")

    while True:
        query = input("Ask a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("👋 Exiting RAG system.")
            break

        retrieved = search(query, index, chunks, k=3)

        answer = generate_answer(query, retrieved)

        print("\n🤖 Generated Answer:\n")
        print(answer)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
