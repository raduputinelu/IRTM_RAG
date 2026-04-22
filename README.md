# IRTM RAG

A Retrieval-Augmented Generation system for question-answering over a psychology textbook, built as a course project for the Information Retrieval & Text Mining course in the AI Master's program at the University of Bucharest.

## What it does

Given selected chapters of an open psychology textbook (Memory, Personality, Psychological Disorders), the system indexes them semantically and answers student questions grounded in the actual textbook text, with page citations. To check whether retrieval actually helps, it also compares against two simpler baselines: keyword-overlap search and the LLM alone with no context.

## Results

| System         | Correct (7 answerable) | Refused off-topic (2) | Refused unrelated (1) |
| -------------- | ---------------------- | --------------------- | --------------------- |
| RAG            | 7/7                    | 2/2                   | 1/1                   |
| Keyword search | 5/7                    | N/A                   | N/A                   |
| LLM only       | 7/7                    | 0/2                   | 0/1                   |

Only the full RAG pipeline answered accurately *and* refused to answer when the information wasn't in the textbook — the behavior needed for educational use.

## Full write-up

Method, results, related work, ethical considerations, references: [RAG_paper_PutineluMihaiRadu.pdf](./RAG_paper_PutineluMihaiRadu.pdf)

## Stack

- Python
- sentence-transformers (MiniLM, 384-d embeddings)
- FAISS
- Llama 3.1 8B via Groq API
- pypdf
- Google Colab

## What I would improve next

- Chunk on paragraph / section boundaries instead of fixed character count
- Adaptive top-k instead of always retrieving 3 chunks
- Add a reranker on top of initial retrieval
- Extend evaluation to multi-step reasoning and multi-turn questions
