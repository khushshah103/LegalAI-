import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []

    def load_documents(self, directory):
        """
        Loads all .txt files from the directory and chunks them.
        """
        self.chunks = []
        self.metadata = []
        
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        for txt_file in txt_files:
            file_path = os.path.join(directory, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple chunking by paragraphs or double newlines
            paragraphs = content.split('\n\n')
            for i, p in enumerate(paragraphs):
                p = p.strip()
                if len(p) > 20: # Ignore very short strings
                    self.chunks.append(p)
                    self.metadata.append({"file": txt_file, "chunk_id": i})
        
        print(f"Loaded {len(self.chunks)} chunks from {len(txt_files)} files.")
        self._build_index()

    def _build_index(self):
        """
        Embeds chunks and builds a FAISS index.
        """
        if not self.chunks:
            return
            
        print("Embedding chunks and building index...")
        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print("Index build successful.")

    def query(self, user_query, top_k=3):
        """
        Searches the index for the most relevant chunks.
        """
        if self.index is None:
            return []
            
        query_embedding = self.model.encode([user_query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
        return results

    def generate_answer(self, query, context_chunks, gemini_client):
        """
        Synthesizes an answer from context chunks using the Gemini LLM.
        """
        if not context_chunks or not gemini_client:
            return "No relevant context found or AI model not connected."
            
        import config
        context_text = "\n\n".join([f"--- Source {i+1} ---\n{c['text']}" for i, c in enumerate(context_chunks)])
        
        prompt = f"""
        Act as a Professional Legal Assistant. Your goal is to answer the user's question with high accuracy, using ONLY the provided legal context chunks.

        [[CONTEXT]]
        {context_text}

        [[INSTRUCTIONS]]
        - Base your answer strictly on the provided context.
        - If the information is missing, state: "I cannot find the answer to this in the document library."
        - Maintain a professional and clear tone.
        - Use bold text for key terms and structured lists if helpful.

        [[QUESTION]]
        {query}
        """
        
        last_error = "Unknown Error"
        for model_id in config.MODEL_FALLBACKS:
            try:
                response = gemini_client.models.generate_content(model=model_id, contents=prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = str(e)
                continue

        return f"Answer generation failed. Last Error: {last_error}"

if __name__ == "__main__":
    # Test script
    import sys
    base_dir = r"c:\Users\User\Downloads\legal document analyzer"
    rag = RAGService()
    rag.load_documents(base_dir)
    
    test_q = "What is the indemnification clause?"
    print(f"\nQuery: {test_q}")
    results = rag.query(test_q)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Source: {res['metadata']['file']}):")
        print(f"{res['text'][:300]}...")
