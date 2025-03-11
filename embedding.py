from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The capital of France is Paris.",
    "France is known for its culture and history.",
    "Paris is a major city in Europe.",
    "The quick brown fox jumps over the lazy dog."
]


embeddings = model.encode(documents, convert_to_tensor=False)
print(f"Embeddings shape: {embeddings.shape}")  


pca = PCA(n_components=2) 
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

for i, doc in enumerate(documents):
    plt.annotate(doc[:20] + "..." if len(doc) > 20 else doc,  
                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 fontsize=8)

plt.title("2D Visualization of Document Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

print("\nSample embedding for the first document:")
print(embeddings[0][:10])  