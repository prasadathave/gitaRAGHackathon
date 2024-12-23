import pandas as pd
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Step 1: Load Datasets

# Load Bhagavad Gita dataset (CSV)
gita_df = pd.read_csv("gita_dataset.csv")  # Replace with the actual dataset path

# Load Mahabharata dataset (CSV)
mahabharata_df = pd.read_csv("mahabharata_dataset.csv")  # Replace with the actual dataset path

# Load Patanjali Yoga Sutras dataset (CSV or other format)
yoga_sutras_df = pd.read_csv("yoga_sutras_dataset.csv")  # Replace with the actual dataset path

# Step 2: Preprocess Data

def preprocess_gita_data(df):
    """
    Combine Sanskrit verses, commentaries, and context for Bhagavad Gita.
    """
    df["retrieval_text"] = df.apply(
        lambda row: f"Sanskrit: {row['Sanskrit']} | Commentary: {row['Swami Adidevananda']} | Question: {row.get('question', '')}",
        axis=1,
    )
    return df

def preprocess_mahabharata_data(df):
    """
    Combine paragraphs and context for Mahabharata.
    """
    df["retrieval_text"] = df.apply(
        lambda row: f"Book {row['Book Number']}, Parva {row['Parva Name']}, Section {row['Section Number']} | {row['Paragraph Text']}",
        axis=1,
    )
    return df

def preprocess_yoga_sutras_data(df):
    """
    Combine sutras and their explanations.
    """
    df["retrieval_text"] = df.apply(
        lambda row: f"Sutra: {row['Sutra']} | Commentary: {row['Commentary']}",
        axis=1,
    )
    return df

# Preprocess all datasets
gita_df = preprocess_gita_data(gita_df)
mahabharata_df = preprocess_mahabharata_data(mahabharata_df)
yoga_sutras_df = preprocess_yoga_sutras_data(yoga_sutras_df)

# Combine all datasets for retrieval
combined_dataset = pd.concat([
    gita_df["retrieval_text"],
    mahabharata_df["retrieval_text"],
    yoga_sutras_df["retrieval_text"]
])

# Step 3: Build Retriever Index

# Create TF-IDF embeddings for the combined dataset
vectorizer = TfidfVectorizer()
corpus_embeddings = vectorizer.fit_transform(combined_dataset)

# Step 4: Define Retrieval Function
def retrieve_text(query, top_k=3):
    """
    Retrieve the top-k most relevant texts for the given query.
    """
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return combined_dataset.iloc[top_indices]

# Step 5: Load Pretrained RAG Components

# Load pretrained tokenizer, retriever, and generator
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path="./passages"  # Placeholder for actual FAISS index and passage paths
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Step 6: Combine Retrieval and Generation

def generate_answer(question):
    """
    Generate an answer to the question using RAG.
    """
    # Retrieve relevant context
    retrieved_texts = retrieve_text(question, top_k=3)

    # Prepare input for generator
    input_text = " " + " \n".join(retrieved_texts.tolist())
    input_ids = tokenizer(
        f"question: {question} context: {input_text}",
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids

    # Generate answer
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Step 7: Example Usage

if __name__ == "__main__":
    question = "What is the teaching about the soul's transmigration in the Gita?"
    answer = generate_answer(question)
    print("Answer:", answer)
