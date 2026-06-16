"""
Script para treinar um modelo de predição de notas baseado em respostas de alunos.

Suporta:
- CSV (formato antigo)
- JSON (novo formato com "answers")

ALTERAÇÃO (validação cruzada para o artigo):
- A avaliação dos modelos agora usa um Pipeline(StandardScaler + regressor), de modo
  que a normalização é ajustada DENTRO de cada dobra da validação cruzada (sem
  vazamento de dados das dobras de teste para o scaler).
- São reportadas, sob validação cruzada k-fold, as métricas: MAE, RMSE, R², QWK
  (Quadratic Weighted Kappa), Pearson e Spearman — calculadas sobre as predições
  out-of-fold — além da média ± desvio-padrão por dobra (robustez).
- As métricas são salvas em models/cv_metrics.csv para citação no artigo.
- O artefato final continua sendo salvo como DOIS arquivos separados
  (semantic_grade_model.pkl e semantic_grade_scaler.pkl), exatamente como a API
  (helpers.load_grade_prediction_model / evaluate.predict_grade) espera.
"""

import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
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
CV_METRICS_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'cv_metrics.csv')

N_SPLITS = 5
RANDOM_STATE = 42

FEATURE_NAMES = [
    "similarity", "length_ratio", "student_words", "student_sentences",
    "keyword_ratio", "avg_word_len", "punct_density",
]


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

        # Define a resposta base como a resposta de maior nota da questão
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
# MÉTRICAS / VALIDAÇÃO CRUZADA
# =========================
def _safe_corr(fn, y_true, y_pred):
    try:
        v = fn(y_true, y_pred)[0]
        return float(v)
    except Exception:  # noqa: BLE001 — entrada degenerada (constante, etc.)
        return float("nan")


def _qwk(y_true, y_pred):
    """QWK com as notas 0..1 binadas em inteiros 0..10."""
    t = np.round(np.clip(y_true, 0, 1) * 10).astype(int)
    p = np.round(np.clip(y_pred, 0, 1) * 10).astype(int)
    try:
        return float(cohen_kappa_score(t, p, weights="quadratic", labels=list(range(11))))
    except Exception:  # noqa: BLE001
        return float("nan")


def oof_metrics(y_true, y_pred):
    """Métricas calculadas sobre as predições out-of-fold (escala 0..1)."""
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "QWK": _qwk(y_true, y_pred),
        "Pearson": _safe_corr(pearsonr, y_true, y_pred),
        "Spearman": _safe_corr(spearmanr, y_true, y_pred),
    }


def cross_validate_models(models, X, y, kf):
    """
    Para cada modelo (já como Pipeline scaler+regressor), calcula:
      - métricas sobre predições out-of-fold (cross_val_predict)
      - média ± desvio-padrão por dobra para MAE, RMSE e R² (robustez)
    Retorna (DataFrame de métricas, dict de predições oof).
    """
    records = []
    oof_predictions = {}

    for name, pipe in models.items():
        # predições out-of-fold: cada amostra é prevista por um modelo que não a viu
        oof = cross_val_predict(pipe, X, y, cv=kf)
        m = oof_metrics(y, oof)

        # média ± desvio por dobra (robustez exigida na metodologia)
        r2_folds = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        mae_folds = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
        rmse_folds = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")

        m.update({
            "MAE_std": float(np.std(mae_folds)),
            "RMSE_std": float(np.std(rmse_folds)),
            "R2_mean_fold": float(np.mean(r2_folds)),
            "R2_std": float(np.std(r2_folds)),
        })
        records.append({"Modelo": name, **m})
        oof_predictions[name] = np.clip(oof, 0, 1)

    cols = ["Modelo", "MAE", "MAE_std", "RMSE", "RMSE_std",
            "R2", "R2_std", "QWK", "Pearson", "Spearman"]
    return pd.DataFrame(records)[cols], oof_predictions


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
    # TUNING (com Pipeline -> scaler ajustado por dobra, sem vazamento)
    # =========================
    print("\n🏋️ Otimizando hiperparâmetros com GridSearchCV (pipeline scaler+modelo)...")

    rf_pipe = Pipeline([("scaler", StandardScaler()),
                        ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
    rf_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
    }
    gs_rf = GridSearchCV(rf_pipe, rf_params, cv=3, scoring="r2", n_jobs=-1)
    gs_rf.fit(X, y)
    best_rf = gs_rf.best_estimator_

    gb_pipe = Pipeline([("scaler", StandardScaler()),
                        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    gb_params = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    }
    gs_gb = GridSearchCV(gb_pipe, gb_params, cv=3, scoring="r2", n_jobs=-1)
    gs_gb.fit(X, y)
    best_gb = gs_gb.best_estimator_

    # candidatos para a validação cruzada (cada um é um Pipeline completo)
    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "RandomForest_Tuned": best_rf,
        "GradientBoosting_Tuned": best_gb,
    }

    # =========================
    # VALIDAÇÃO CRUZADA (k-fold) — métricas para o artigo
    # =========================
    print(f"\n🔁 Validação cruzada {N_SPLITS}-fold (métricas out-of-fold)...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics_df, oof_preds = cross_validate_models(models, X, y, kf)

    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print("\n" + metrics_df.to_string(index=False))

    # salva métricas para citação
    os.makedirs(os.path.dirname(CV_METRICS_PATH), exist_ok=True)
    metrics_df.to_csv(CV_METRICS_PATH, index=False)
    print(f"\n💾 Métricas de validação cruzada salvas em: {CV_METRICS_PATH}")

    # seleção pelo melhor R² médio por dobra (critério original, agora sem vazamento)
    best_idx = metrics_df["R2"].idxmax()
    best_model_name = metrics_df.loc[best_idx, "Modelo"]
    print(f"\n🏆 Modelo vencedor: {best_model_name} "
          f"(MAE={metrics_df.loc[best_idx, 'MAE']:.4f}, "
          f"RMSE={metrics_df.loc[best_idx, 'RMSE']:.4f}, "
          f"QWK={metrics_df.loc[best_idx, 'QWK']:.4f})")

    # =========================
    # AJUSTE FINAL EM TODOS OS DADOS + SALVAMENTO (scaler e modelo SEPARADOS)
    # =========================
    best_pipeline = models[best_model_name]
    best_pipeline.fit(X, y)  # garante scaler e modelo ajustados em todo o dataset
    final_scaler = best_pipeline.named_steps["scaler"]
    final_model = best_pipeline.named_steps["model"]

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(final_scaler, f)

    print(f"💾 Modelo ({best_model_name}) e Scaler salvos separadamente "
          f"(compatível com a API).")
    return True


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Treinador de Modelo Semântico (com validação cruzada)")
    print("=" * 60)

    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    success = train_model(DATA_PATH)
    sys.exit(0 if success else 1)
