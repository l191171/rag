from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import re
from transformers import pipeline
import os

# Step 1: Preprocessing
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def preprocess_text(text, max_chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_chunk_size - overlap
    return chunks

# Step 2: Initialize Models chn
# Choose a legal-specific model if available
model = SentenceTransformer('all-MPNet-base-v2')  # Replace with a legal-specific model if possible 
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # For reranking
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Step 3: Initialize ChromaDB with Persistence
persist_path = "chroma_db"  # Specify your persistence path here
chroma_client = chromadb.PersistentClient(path=persist_path)

# Check if the collection already exists
collection_name = "my_collection2"
# # try:
chroma_client.delete_collection(name=collection_name)
# collection = chroma_client.create_collection(name=collection_name)
flag=0
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
    flag = 1
    # Optionally, you might want to renew it by deleting and recreating
    # chroma_client.delete_collection(name=collection_name)
    # collection = chroma_client.create_collection(name=collection_name)
except Exception as e:
    print(f"Creating new collection '{collection_name}'")
    collection = chroma_client.create_collection(name=collection_name)  

from langchain_community.document_loaders import TextLoader  # Updated import///]]/
DATA_PATH = "cases"
# Step 4: Add Documents
long_documents = []
target_categories = {'CLC', 'CLCN', 'CLD', 'GBLR', 'MLD', 'PCRLJ', 'PCRLJN', 'PLC', 'PLC(CS)', 'PLC(CS)N', 'PLCN', 'PLD', 'PTD', 'SCMR', 'YLR', 'YLRN'}  # Include multiple categories
target_years = {'2023', '2024'}  # Include multiple years

# Loop through each target category
for category in target_categories:
    category_path = os.path.join(DATA_PATH, category)
    if os.path.isdir(category_path):
        for year_folder in os.listdir(category_path):
            if year_folder in target_years:
                year_path = os.path.join(category_path, year_folder)
                if os.path.isdir(year_path):
                    for case_folder in os.listdir(year_path):
                        case_path = os.path.join(year_path, case_folder)
                        if os.path.isdir(case_path):
                            for file_name in os.listdir(case_path):
                                file_path = os.path.join(case_path, file_name)
                                if file_path.endswith("headnotes.txt"):
                                    try:
                                        # Load text content using TextLoader
                                        loader = TextLoader(file_path, encoding='utf-8')
                                        loaded_docs = loader.load()
                                        # Extract the text content from loaded documents
                                        long_documents.extend([doc.page_content for doc in loaded_docs])
                                    except UnicodeDecodeError as e:
                                        print(f"Error decoding {file_path}: {e}")
                                    except Exception as e:
                                        print(f"Error loading {file_path}: {e}")

if flag == 0:
    for doc_id, long_document in enumerate(long_documents):
        cleaned_text = clean_text(long_document)
        chunks = preprocess_text(cleaned_text)
        embeddings = model.encode(chunks)
        
        print("embeddings:")
        print(embeddings)
        print("chunks:")
        print(chunks)
        print("cleaned_text:")
        print(cleaned_text)

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "source": f"Document_{doc_id}",
                    "chunk_number": i,
                }
                for i in range(len(chunks))
            ]
        )

        # print(collection)

# Persist the collection if needed
# chroma_client.persist()

def query_system(query_text, model, cross_encoder, qa_pipeline, collection, n_results=5):
    # Embed the query
    print(f"Number of documents in collection: {collection.count()}")

    query_embedding = model.encode([query_text])
    
    # Retrieve top N chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    if not results['documents']:
        print("No documents retrieved.")
        return {"answer": "No relevant documents found.", "score": 0, "relevant_chunks": []}

    retrieved_chunks = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]

    # Check what has been retrieved
    print(f"Retrieved Chunks: {retrieved_chunks}")
    print(f"Retrieved Metadata: {retrieved_metadata}")
    
    # Rerank using cross-encoder
    query_doc_pairs = [(query_text, doc) for doc in retrieved_chunks]
    if not query_doc_pairs:
        print("No document pairs created for cross-encoder.")
        return {"answer": "No relevant document pairs to rerank.", "score": 0, "relevant_chunks": []}
    
    scores = cross_encoder.predict(query_doc_pairs)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranked_chunks = [retrieved_chunks[i] for i in sorted_indices]
    
    # Answer the question using QA model
    best_answer = answer_question(query_text, reranked_chunks[:5])  # Use top 3 reranked chunks
    
    return {
        "answer": best_answer['answer'],
        "score": best_answer['score'],
        "relevant_chunks": [reranked_chunks[i] for i in range(min(5, len(reranked_chunks)))]
    }

# Step 6: Answer Question
def answer_question(query_text, retrieved_chunks):
    answers = []
    for chunk in retrieved_chunks:
        try:
            answer = qa_pipeline(question=query_text, context=chunk)
            answers.append(answer)
        except Exception as e:
            print(f"Error processing chunk: {e}")
    if answers:
        best_answer = max(answers, key=lambda x: x['score'])
        return best_answer
    else:
        return {"answer": "No answer found.", "score": 0}

# Example Usage
if __name__ == "__main__":
    query_text = "Land Acquisition dispute of Azad Jammu and Kashmir"
    response = query_system(query_text, model, cross_encoder, qa_pipeline, collection, n_results=5)
    
    print(f"Answer: {response['answer']}")
    print(f"Score: {response['score']}")
    print("\nRelevant Chunks:")
    for i, chunk in enumerate(response['relevant_chunks']):
        print(f"Chunk {i+1}: {chunk[:500]}...\n")
