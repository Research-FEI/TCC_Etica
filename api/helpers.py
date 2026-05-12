
import re
import os
import pickle
import torch
from flask import json
from transformers import AutoTokenizer, AutoModel

# Carrega o modelo uma única vez para reutilizá-lo
_tokenizer = None
_bert_model = None
_grade_predictor = None
_grade_scaler = None
_questions = {"questions": []}

#Load questions from JSON
questions_path = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'questions.json')
try:
    with open(questions_path, 'r', encoding='utf-8') as f:
        _questions = json.load(f)
    print(f"Questões carregadas com sucesso! Total: {len(_questions.get('questions', []))}")
except FileNotFoundError:
    _questions = {"questions": []}
    print(f"Arquivo de questões não encontrado em {questions_path}")

print("Modelo de IA será carregado na primeira requisição de avaliação...")

def get_model():
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        model_name = "bert-base-multilingual-cased"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bert_model = AutoModel.from_pretrained(model_name)
    return _tokenizer, _bert_model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_bert_embeddings(texts, tokenizer, model):
    # Tokenize input texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings.numpy()

def load_grade_prediction_model():
    global _grade_predictor, _grade_scaler
    
    if _grade_predictor is None:
        model_path = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_scaler.pkl')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    _grade_predictor = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    _grade_scaler = pickle.load(f)
                print("✅ Modelo de predição de grades carregado!")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de grades: {e}")
    
    return _grade_predictor, _grade_scaler


### Done - normalize text removing extra spaces and converting to lowercase
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

### Done - concatenate feedback from missing keywords and semantic similarity into a single string
def concatanate_feedback(missing_keywords, semantic_similarity_feedback):
    feedback = []
    if not missing_keywords or not semantic_similarity_feedback:
        return "Unable to generate feedback"
    else:
        feedback.append(f"Faltaram os seguintes tópicos: {', '.join(missing_keywords)}.")
        feedback.append(f"O que faltou para melhorar a nota: {semantic_similarity_feedback}.")
    return " ".join(feedback)