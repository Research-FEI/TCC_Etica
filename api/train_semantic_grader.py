"""
Script para treinar um modelo de predição de grades baseado em respostas de alunos

Agora suporta:
- CSV (formato antigo)
- JSON (novo formato com "answers")
"""

import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from helpers import get_model, get_bert_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# =========================
# CONFIG
# =========================
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'answers.json')
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'questions.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_scaler.pkl')


# =========================
# LOADERS
# =========================
def load_csv(path):
    df = pd.read_csv(path)

    required_cols = ['Question', 'Base_answer', 'Student_answer', 'Grade']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV deve ter colunas: {required_cols}")

    # Scale Grade to 0-1 if it's currently 0-10
    if df['Grade'].max() > 1.0:
        df['Grade'] = df['Grade'] / 10.0

    return df


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    for question in data["answers"]:
        qid = question["id"]

        # 🔥 Definindo resposta base automaticamente (primeira resposta com maior nota)
        answers_sorted = sorted(question["answers"], key=lambda x: x["grade"], reverse=True)
        base_answer = answers_sorted[0]["answer"]

        for ans in question["answers"]:
            grade = float(ans["grade"])
            # Scale Grade to 0-1 if it's currently 0-10
            if grade > 1.0:
                grade = grade / 10.0
                
            rows.append({
                "Question": qid,
                "Base_answer": base_answer,
                "Student_answer": ans["answer"],
                "Grade": grade
            })

    return pd.DataFrame(rows)


def load_data(path):
    if path.endswith(".csv"):
        print("📄 Carregando CSV...")
        return load_csv(path)
    elif path.endswith(".json"):
        print("📄 Carregando JSON...")
        return load_json(path)
    else:
        raise ValueError("Formato não suportado. Use CSV ou JSON.")


def load_questions():
    if not os.path.exists(QUESTIONS_PATH):
        return {}
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {q['id']: q for q in data.get('questions', [])}


# =========================
# MODEL
# =========================
def load_semantic_transformer():
    return get_model()


def extract_features(student_answer, base_answer, similarity, keywords=None):
    # Word counts
    student_tokens = student_answer.split()
    student_words = len(student_tokens) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0

    # Sentence counts
    student_sentences = max(1, len([s for s in re.split(r"[\.\!?]+", student_answer) if s.strip()]))
    
    # Length ratio
    length_ratio = student_words / base_words if base_words > 0 else 0

    # Keyword overlap
    keyword_ratio = 0
    if keywords and student_answer:
        found = 0
        for kw in keywords:
            if kw.lower() in student_answer.lower():
                found += 1
        keyword_ratio = found / len(keywords)

    # Avg word length
    avg_word_len = np.mean([len(w) for w in student_tokens]) if student_tokens else 0

    # Punctuation density
    punct_count = len(re.findall(r'[^\w\s]', student_answer))
    punct_density = punct_count / len(student_answer) if student_answer else 0

    return np.array([
        similarity,
        length_ratio,
        student_words,
        student_sentences,
        keyword_ratio,
        avg_word_len,
        punct_density
    ])


# =========================
# TRAIN
# =========================
def train_model(data_path):

    if not os.path.exists(data_path):
        print(f"❌ Arquivo não encontrado: {data_path}")
        return False

    print("📚 Carregando dados...")
    try:
        df = load_data(data_path)
        questions_map = load_questions()
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False

    print(f"✅ {len(df)} amostras carregadas")

    df = df.dropna(subset=['Base_answer', 'Student_answer', 'Grade'])

    if len(df) < 10:
        print("❌ Poucos dados para treino (mínimo 10)")
        return False

    print("🤖 Carregando modelo de embeddings...")
    tokenizer, model = load_semantic_transformer()

    print("🔍 Extraindo features...")
    X = []
    y = []

    for idx, row in df.iterrows():
        try:
            student = str(row['Student_answer'])
            base = str(row['Base_answer'])
            grade = float(row['Grade'])
            qid = row['Question']

            keywords = questions_map.get(qid, {}).get('keywords', [])

            emb = get_bert_embeddings([student, base], tokenizer, model)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

            features = extract_features(student, base, sim, keywords)

            X.append(features)
            y.append(grade)

        except Exception as e:
            print(f"⚠️ Erro linha {idx}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n📊 Estatísticas das Notas:")
    print(f"Min: {y.min():.4f} | Max: {y.max():.4f} | Média: {y.mean():.4f}")

    # =========================
    # SCALE
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # MODELS COMPARISON & TUNING
    # =========================
    print("\n🏋️ Otimizando modelos com GridSearchCV...")
    
    # Grid search for Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    gs_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
    gs_rf.fit(X_scaled, y)
    best_rf = gs_rf.best_estimator_
    
    # Grid search for Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gs_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='r2', n_jobs=-1)
    gs_gb.fit(X_scaled, y)
    best_gb = gs_gb.best_estimator_

    models = {
        "Ridge": Ridge(),
        "RandomForest_Tuned": best_rf,
        "GradientBoosting_Tuned": best_gb
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_r2 = -np.inf
    best_model_name = ""

    for name, m in models.items():
        # Usamos R² para seleção, mas acompanhamos MAE
        r2_scores = cross_val_score(m, X_scaled, y, cv=kf, scoring='r2')
        mae_scores = cross_val_score(m, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
        
        avg_r2 = np.mean(r2_scores)
        avg_mae = -np.mean(mae_scores)
        
        print(f"  - {name}: R²={avg_r2:.4f}, MAE={avg_mae:.4f}")

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model_name = name

    print(f"\n🏆 Modelo vencedor: {best_model_name}")

    # Final fit with all data using the best model
    final_model = models[best_model_name]
    final_model.fit(X_scaled, y)

    # =========================
    # SAVE
    # =========================
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"💾 Modelo {best_model_name} e Scaler salvos com sucesso!")
    return True


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Treinador de Modelo Semântico")
    print("=" * 60)

    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    success = train_model(DATA_PATH)
    sys.exit(0 if success else 1)