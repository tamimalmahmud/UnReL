from typing import List, Dict
import os
import re
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import time

if __name__ == "__main__":
    start_time = time.time() 

class AutoScoringForgetProbability:
    def __init__(self, device='cuda'):
        # Updated weights
        self.weights = {
            'data_category': 0.55,  # How sensitive the topic/content is
            'legal_risk': 0.25,     # Legal importance (e.g., PII, medical)
            'freshness': 0.15,      # Recency of data (based on year)
            'source_type': 0.10     # Origin/public nature of the data
        }

        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        self.category_texts = {
            'medical': ['medical', 'health', 'patient', 'diagnosis', 'treatment'],
            'finance': ['finance', 'bank', 'investment', 'revenue', 'financial report'],
            'personal': ['personal', 'identity', 'name', 'address', 'social security'],
            'generic': ['general knowledge', 'encyclopedia', 'news', 'Wikipedia', 'open source', 'Python', 'programming', 'software']
        }
        self.category_embeddings = {
            k: self.model.encode(v, convert_to_tensor=True).mean(0)
            for k, v in self.category_texts.items()
        }

    def _score_data_category(self, text: str) -> float:
        """Evaluate sensitivity of the topic/content"""
        text_emb = self.model.encode(text, convert_to_tensor=True)
        sensitive_emb = torch.stack([self.category_embeddings['personal'], self.category_embeddings['medical']])
        sims = util.cos_sim(text_emb, sensitive_emb)
        max_sim = sims.max().item()
        # Boost for strong PII patterns
        if re.search(r'\bJohn Doe\b|\d{3}-\d{2}-\d{4}|\d{3} \w+ Street\b', text):
            max_sim = min(1.0, max_sim + 0.2)
        return float(max(0.3, min(1.0, max_sim)))

    def _score_legal_risk(self, dataset: Dict) -> float:
        """Legal risk (PII, medical, etc.)"""
        risk = dataset.get('legal_risk', 'medium').lower()
        if risk == 'high': return 1.0
        if risk == 'medium': return 0.5
        return 0.1

    def _score_freshness(self, text: str) -> float:
        """Freshness based on years in text"""
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]
        if not years:
            return 0.6
        max_year = max(years)
        age = 2025 - max_year
        if age < 1:
            return 1.0
        elif age <= 3:
            return 0.85
        elif age <= 5:
            return 0.7
        else:
            return 0.5

    def _score_source_type(self, text: str) -> float:
        """Source type: public vs private"""
        lower_text = text.lower()
        if any(w in lower_text for w in ['wikipedia', 'open source', 'github', 'public']):
            return 0.2
        if any(w in lower_text for w in ['internal', 'confidential', 'private', 'restricted']):
            return 1.0
        return 0.5

    def _weighted_probability(self, dataset: Dict, text: str) -> float:
        S_data = self._score_data_category(text)
        S_legal = self._score_legal_risk(dataset)
        S_fresh = self._score_freshness(text)
        S_source = self._score_source_type(text)

        P_forget = (
            self.weights['data_category'] * S_data +
            self.weights['legal_risk'] * S_legal +
            self.weights['freshness'] * S_fresh +
            self.weights['source_type'] * S_source
        )
        return round(max(0.0, min(1.0, P_forget)), 3)

    def calculate_for_local_dataset(self, datasets_dir: str) -> List[Dict]:
        results = []

        for filename in os.listdir(datasets_dir):
            filepath = os.path.join(datasets_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == '.txt':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    prob = self._weighted_probability({'legal_risk': 'medium'}, content)
                    results.append({
                        'file': filename,
                        'probability': prob,
                        'text_preview': content[:200] + '...' if len(content) > 200 else content
                    })

                elif ext == '.csv':
                    df = pd.read_csv(filepath)
                    combined_text = " ".join([f"{row.get('question','')} {row.get('answer','')}" for _, row in df.iterrows()])
                    prob = self._weighted_probability({'legal_risk': 'medium'}, combined_text)
                    results.append({
                        'file': filename,
                        'probability': prob,
                        'text_preview': combined_text[:200] + '...' if len(combined_text) > 200 else combined_text
                    })

                elif ext == '.json':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, dict):
                                data = [data]
                        except json.JSONDecodeError:
                            f.seek(0)
                            data = [json.loads(line) for line in f if line.strip()]

                    combined_text = " ".join([f"{entry.get('question','')} {entry.get('answer','')}" for entry in data])
                    prob = self._weighted_probability({}, combined_text)
                    results.append({
                        'file': filename,
                        'probability': prob,
                        'text_preview': combined_text[:200] + '...' if len(combined_text) > 200 else combined_text
                    })

            except Exception as e:
                print(f"Error reading {filename}: {e}")

        return results


if __name__ == "__main__":
    start_time = time.time()  # Start timer
    calculator = AutoScoringForgetProbability()
    datasets_dir = "./CalculateForgetProbabilityData/forgetdata"  # your folder with txt/json/csv
    results = calculator.calculate_for_local_dataset(datasets_dir)

    for r in results:
        print(f"File: {r['file']} | Probability: {r['probability']} | Preview: {r['text_preview'][:60]}...")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")    
    with open(os.path.join(datasets_dir, 'running_time.txt'), 'w') as f:
        f.write(f"Total running time: {elapsed_time:.2f} seconds\n")
