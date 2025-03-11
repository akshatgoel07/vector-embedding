from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = SentenceTransformer('all-MiniLM-L6-v2')

long_document = """
The capital of France is Paris, a city renowned for its culture, art, and history. 
France is known for its rich heritage, including landmarks like the Eiffel Tower and the Louvre Museum. 
Paris, often called the City of Light, attracts millions of tourists annually due to its architecture and cuisine. 
The quick brown fox jumps over the lazy dog, a common phrase used to test typing skills. 
France also has a significant role in European politics and economy, with Paris being a major global hub.
"""

def chunk_text(text, max_sentences=2):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    sentence_count = 0
    
    for sentence in sentences:
        if not sentence:
            continue
        if sentence_count < max_sentences:
            current_chunk += sentence + ". "
            sentence_count += 1
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
            sentence_count = 1
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(long_document, max_sentences=2)
print("Generated chunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")

embeddings = model.encode(chunks, convert_to_tensor=False)
print(f"Embeddings shape: {embeddings.shape}")

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

for i, chunk in enumerate(chunks):
    plt.annotate(chunk[:20] + "..." if len(chunk) > 20 else chunk,
                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 fontsize=8)

plt.title("2D Visualization of Chunked Document Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

print("\nSample embedding for the first chunk:")
print(embeddings[0][:10])
