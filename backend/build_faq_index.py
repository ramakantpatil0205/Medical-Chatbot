"""Build embeddings index from a CSV of FAQs using sentence-transformers and FAISS."""
import argparse, os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

def main(input_csv, output_dir, model_name='all-MiniLM-L6-v2'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    texts = (df['question'].fillna('') + ' ' + df['answer'].fillna('')).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, 'faqs.faiss'))
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'questions': df['question'].tolist(), 'answers': df['answer'].tolist(), 'source': df['source'].tolist()}, f)
    print('Index saved to', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '--input_csv', dest='input_csv', required=True)
    parser.add_argument('--output', dest='output_dir', required=True)
    args = parser.parse_args()
    main(args.input_csv, args.output_dir)
