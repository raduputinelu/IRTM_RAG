!pip install pypdf sentence-transformers faiss-cpu google-generativeai groq -q

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from google.colab import userdata

pdf_path = "/content/drive/MyDrive/Colab Notebooks/Psychology2e_WEB.pdf"
reader = PdfReader(pdf_path)

# Chapter 8 Memory, Chapter 12 Social psychology and Chapter 15 Psychological disorders
pages_to_use = list(range(246, 277)) + list(range(398, 444)) + list(range(536, 598))

documents = []
for page_num in pages_to_use:
    text = reader.pages[page_num].extract_text()
    if text.strip():
        documents.append({
            "text": text,
            "page": page_num + 1
        })

def split_into_chunks(documents, chunk_size=1000, overlap=200):
    chunks = []

    for doc in documents:
        text = doc["text"]
        page = doc["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page
                })

            start = end - overlap

    return chunks

chunks = split_into_chunks(documents)

model = SentenceTransformer('all-MiniLM-L6-v2')

chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(chunk_texts, show_progress_bar=True)

embeddings_array = np.array(embeddings).astype('float32')

dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

def retrieve(query, top_k=3):
    q_embed = model.encode([query])
    q_embed = np.array(q_embed).astype('float32')

    distances, indices = index.search(q_embed, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "distance": distances[0][i]
        })

    return results

results = retrieve("What is memory?")

for i, r in enumerate(results):
    print(f"Result {i+1} (page {r['page']}, distance: {r['distance']:.2f})")
    print(r["text"][:300] + "\n")

GROQ_API_KEY = userdata.get('GROQ_API_KEY')

client = Groq(api_key=GROQ_API_KEY)

def ask(question, top_k=3):
    retrieved = retrieve(question, top_k)

    context = ""
    source_pages = []
    for chunk in retrieved:
        context += chunk["text"] + "\n\n"
        source_pages.append(chunk["page"])

    prompt = f"""Answer based on this context. If the answer isn't in the context, say so.

Context:{context}

Question:{question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": source_pages
    }

result = ask("What is short-term memory?")

print("Answer:", result["answer"])
print("Source pages:", result["sources"])

def ir_search(query, top_k=3):
    query_words = set(query.lower().split())

    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk["text"].lower().split())

        matches = len(query_words.intersection(chunk_words))

        scored.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": matches
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]

results = ir_search("What is short-term memory?")

for i, r in enumerate(results):
    print(f"Result {i+1} (page {r['page']}, matches: {r['score']})")
    print(r["text"][:300] + "\n")

def ask_llm(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content

llm_answer = ask_llm("What is short-term memory?")
print("LLM answer:", llm_answer)

def compare_all(question):
    print(f"Question: {question}")

    rag_result = ask(question)
    print(f"RAG Answer: {rag_result['answer']}")
    print(f"RAG Source pages: {rag_result['sources']}\n")

    ir_results = ir_search(question)
    print(f"IR top result (page {ir_results[0]['page']}):")
    print(ir_results[0]["text"][:200] + "\n")

    llm_answer = ask_llm(question)
    print(f"LLM answer: {llm_answer}")

compare_all("What are the stages of memory?")

queries_with_answers = [
    "What is short-term memory?",
    "What are the stages of memory?",
    "What is classical conditioning?",
    "What are the Big Five personality traits?",
    "What is schizophrenia?",
    "What is the difference between retrograde and anterograde amnesia?",
    "What is Freud's theory of personality?"
]
queries_no_answer = [
    "What is photosynthesis?",
    "How does the stock market work?"
]
query_unrelated = "What is the capital of France?"

print("QUERIES WITH EXPECTED ANSWERS")

for i, q in enumerate(queries_with_answers, 1):
    print(f"Query{i}: {q}")
    result = ask(q)
    print(f"Answer: {result['answer'][:500]}...")
    print(f"Sources: pages {result['sources']}")

print("QUERIES WITH NO EXPECTED ANSWERS")

for i, q in enumerate(queries_no_answer, 1):
    print(f"Query{i}: {q}")
    result = ask(q)
    print(f"Answer: {result['answer']}")
    print(f"Sources: pages {result['sources']}")

print("GENERAL UNRELATED QUERY")

print(f"Query: {query_unrelated}")

result = ask(query_unrelated)
print(f"Answer: {result['answer']}")
print(f"Sources: pages {result['sources']}")
