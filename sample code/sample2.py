# Install required libraries
# pip install transformers sentence-transformers faiss-cpu pandas

import pandas as pd
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sentence_transformers import SentenceTransformer
import faiss

# Load the dataset (adjust the file paths as needed)
bhagavad_gita_df = pd.read_csv("bhagavad_gita.csv")  # Replace with your CSV
mahabharata_df = pd.read_csv("mahabharata.csv")  # Replace with your Mahabharata dataset
concept_mapping_df = pd.read_csv("concept_mapping.csv")  # Concept mapping CSV

# Preprocess data into a unified corpus
def preprocess_data():
    corpus = []

    # Process Mahabharata
    for _, row in mahabharata_df.iterrows():
        text = f"Book {row['Book Number']}, {row['Parva Name']} - {row['Paragraph Text']}"
        corpus.append(text)

    # Process Bhagavad Gita with commentaries
    for _, row in bhagavad_gita_df.iterrows():
        text = f"Chapter {row['Chapter']}, Verse {row['Verse']} ({row['Speaker']}): {row['Sanskrit']} - {row['Swami Gambirananda']}"
        corpus.append(text)

    # Process Concept Mapping
    for _, row in concept_mapping_df.iterrows():
        text = f"Concept: {row['Concept']} - {row['Sanskrit']} ({row['English']})"
        corpus.append(text)

    return corpus

corpus = preprocess_data()

# Step 1: Generate embeddings using Sentence Transformers
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)

# Step 2: Build a FAISS index for fast retrieval
embedding_dim = corpus_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(corpus_embeddings.cpu().numpy())

# Step 3: Set up the RAG retriever and generator
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
rag_retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages=corpus,
    index_path=None
)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Function to find top passages using FAISS
def retrieve_passages(question, top_k=5):
    question_embedding = embedding_model.encode([question], convert_to_tensor=True)
    distances, indices = faiss_index.search(question_embedding.cpu().numpy(), top_k)
    results = [corpus[idx] for idx in indices[0]]
    return results

# Function to generate answers using RAG
def generate_answer(question):
    # Retrieve top passages
    top_passages = retrieve_passages(question, top_k=5)
    context = " ".join(top_passages)

    # Generate an answer
    inputs = rag_tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    gen_outputs = rag_model.generate(
        input_ids,
        context_input_ids=rag_tokenizer(context, return_tensors="pt")["input_ids"]
    )
    answer = rag_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
    return answer[0], top_passages

# Example usage
question = "What does Bhagavad Gita say about the soul?"
answer, retrieved_passages = generate_answer(question)

print("Question:", question)
print("Answer:", answer)
print("Retrieved Passages:")
for passage in retrieved_passages:
    print("-", passage)
