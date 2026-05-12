"""
Script para treinar um modelo de predição de grades baseado em respostas de alunos

Agora suporta:
- CSV (formato antigo)
- JSON (novo formato com "answers")
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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


# =========================
# MODEL
# =========================
def load_semantic_transformer():
    return get_model()


def extract_features(student_answer, base_answer, similarity):
    student_words = len(student_answer.split()) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0

    student_sentences = max(1, len([s for s in student_answer.split('.') if s.strip()]))
    length_ratio = student_words / base_words if base_words > 0 else 0

    return np.array([
        similarity,
        length_ratio,
        student_words,
        student_sentences,
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

            emb = get_bert_embeddings([student, base], tokenizer, model)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

            features = extract_features(student, base, sim)

            X.append(features)
            y.append(grade)

        except Exception as e:
            print(f"⚠️ Erro linha {idx}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n📊 Estatísticas:")
    print(f"Min: {y.min():.2f} | Max: {y.max():.2f} | Média: {y.mean():.2f}")

    # =========================
    # SCALE
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # =========================
    # MODELS
    # =========================
    print("\n🏋️ Treinando...")

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    r_pred = ridge.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    r2_ridge = r2_score(y_test, r_pred)
    r2_rf = r2_score(y_test, rf_pred)

    mae_ridge = mean_absolute_error(y_test, r_pred)
    mae_rf = mean_absolute_error(y_test, rf_pred)

    if r2_rf > r2_ridge:
        model = rf
        print(f"🏆 RandomForest (R²={r2_rf:.4f}, MAE={mae_rf:.4f})")
    else:
        model = ridge
        print(f"🏆 Ridge (R²={r2_ridge:.4f}, MAE={mae_ridge:.4f})")

    # =========================
    # SAVE
    # =========================
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("\n💾 Modelo salvo com sucesso!")
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