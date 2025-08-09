"""Simple RAG-style chat backend.
- Loads FAISS index + metadata
- Retrieves top-k relevant FAQ entries
- (Placeholder) Combines retrieved passages and forms an answer using a simple template.
Replace the `generate_answer` function with an LLM call (OpenAI/transformers) for production.
"""
import os, pickle, faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGChat:
    def __init__(self, index_dir='indexes', model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(os.path.join(index_dir, 'faqs.faiss'))
        with open(os.path.join(index_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.questions = meta['questions']
        self.answers = meta['answers']
        self.sources = meta['source']
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query, topk=3):
        qvec = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(qvec, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({'question': self.questions[idx], 'answer': self.answers[idx], 'source': self.sources[idx], 'score': float(score)})
        return results

    def generate_answer(self, query, retrieved):
        # Simple synthesis: show concise answer formed from top retrieved entries.
        synth = ' '.join([r['answer'] for r in retrieved])
        response = f"Based on matching FAQs, here's a concise answer:\n\n{synth}\n\nSources: {', '.join(set([r['source'] for r in retrieved]))}"
        return response

    def answer(self, query, topk=3):
        retrieved = self.retrieve(query, topk=topk)
        answer = self.generate_answer(query, retrieved)
        return answer, retrieved

if __name__ == '__main__':
    rc = RAGChat(index_dir='indexes')
    ans, retrieved = rc.answer('What are flu symptoms?')
    print(ans)
