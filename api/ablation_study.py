#!/usr/bin/env python3
"""
ablation_study.py
==================
Estudo de ablação para o artigo: mede o impacto de cada componente do sistema
na qualidade da predição, usando validação cruzada k-fold (mesmo protocolo do
train_semantic_grader.py, com o scaler ajustado DENTRO de cada fold via
Pipeline -- sem vazamento de dados).

Roda DOIS tipos de ablação:

  (A) FEATURES ESTRUTURAIS INDIVIDUAIS
      Para cada feature do vetor de atributos (similarity, length_ratio,
      student_words, student_sentences, keyword_ratio, avg_word_len,
      punct_density), treina o modelo SEM essa feature (mantendo as outras)
      e compara contra o modelo completo. A queda de desempenho ao remover
      uma feature indica sua contribuição.

  (B) COMPONENTES DO SCORE HÍBRIDO
      Reproduz a Composição Híbrida da Nota descrita na Seção III-D do TCC
      (60% regressão semântico-estrutural + 40% cobertura de palavras-chave)
      e compara quatro variantes:
        - "Apenas semântico-estrutural": só a saída do regressor,
          sem o componente de palavras-chave (peso 100/0)
        - "Apenas palavras-chave": só keyword_ratio, sem o regressor (peso 0/100)
        - "Apenas similaridade de cosseno": só o score bruto do BERT, sem
          regressor e sem keywords (baseline adicional)
        - "Híbrido completo (60/40)": a composição usada no sistema final

REGRESSOR USADO NO COMPONENTE SEMÂNTICO-ESTRUTURAL:
    O regressor usado aqui é o GradientBoostingRegressor, tunado por
    GridSearchCV com a MESMA grade de hiperparâmetros e o MESMO protocolo
    (Pipeline scaler+modelo, CV interna de 3 folds, scoring="r2") definidos
    em train_semantic_grader.py. Isso é proposital: o GridSearchCV em
    train_semantic_grader.py seleciona automaticamente o melhor modelo entre
    Ridge, RandomForest e GradientBoosting pelo R² médio em validação cruzada,
    e o GradientBoosting tunado venceu essa comparação (ver cv_metrics.csv) --
    é também o modelo de fato serializado em
    Service/models/semantic_grade_model.pkl e usado em produção. Uma versão
    anterior deste script usava um RandomForestRegressor não tunado
    (n_estimators=200, sem GridSearchCV) nesta etapa, o que tornava a
    ablação inconsistente com o modelo realmente treinado/implantado. Essa
    inconsistência foi corrigida aqui.

IMPORTANTE: este script reusa a MESMA extração de features e embeddings do
train_semantic_grader.py (mesmo BERT/sentence-transformer, mesmo dataset),
para que as comparações sejam justas. Ele importa `extract_features`,
`load_data`, `load_questions` e `load_semantic_transformer` diretamente do
módulo de treino -- rode-o de dentro de api/ ou ajuste o sys.path.

Uso (a partir da pasta api/, ou da raiz com o caminho ajustado):
    python ablation_study.py
    python ablation_study.py --data Service/data/answers.json --outdir ../resultados_ablation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

# reaproveita as funções já validadas do script de treino principal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train_semantic_grader import (
        load_data, load_questions, load_semantic_transformer, extract_features,
        FEATURE_NAMES,
    )
except ImportError:
    sys.exit(
        "❌ Não foi possível importar train_semantic_grader.py.\n"
        "   Rode este script a partir da mesma pasta (api/), ou ajuste o sys.path."
    )

N_SPLITS = 5
RANDOM_STATE = 42

# índice de cada feature dentro do vetor retornado por extract_features():
# [similarity, length_ratio, student_words, student_sentences,
#  keyword_ratio, avg_word_len, punct_density]
SIMILARITY_IDX = FEATURE_NAMES.index("similarity")
KEYWORD_IDX = FEATURE_NAMES.index("keyword_ratio")


# --------------------------------------------------------------------------- #
# Métricas (mesmas definições do train_semantic_grader.py / metrics_benchmark.py)
# --------------------------------------------------------------------------- #
def _qwk(y_true, y_pred):
    t = np.round(np.clip(y_true, 0, 1) * 10).astype(int)
    p = np.round(np.clip(y_pred, 0, 1) * 10).astype(int)
    try:
        return float(cohen_kappa_score(t, p, weights="quadratic", labels=list(range(11))))
    except Exception:  # noqa: BLE001
        return float("nan")


def _safe_corr(fn, y_true, y_pred):
    try:
        return float(fn(y_true, y_pred)[0])
    except Exception:  # noqa: BLE001
        return float("nan")


# --------------------------------------------------------------------------- #
# Regressor tunado: GradientBoosting via GridSearchCV
# (mesma grade e protocolo de train_semantic_grader.py, para que o
# "componente semântico-estrutural" deste ablation use o MESMO modelo que
# de fato venceu a seleção e foi salvo em produção)
# --------------------------------------------------------------------------- #
def tuned_gb_pipeline(X: np.ndarray, y: np.ndarray, question_ids: np.ndarray) -> Pipeline:
    """Roda o GridSearchCV do GradientBoosting (mesma grade do
    train_semantic_grader.py) e retorna o melhor Pipeline (scaler + modelo),
    ainda NÃO ajustado nos dados completos -- cross_val_predict cuida do
    fit/predict por fold internamente.

    A CV interna do GridSearchCV (3 folds) é estratificada por `question_ids`,
    pelo mesmo motivo da CV externa em train_semantic_grader.py: evita que o
    tuning escolha hiperparâmetros ajustados a uma composição de fold
    enviesada por questão, dado o desempenho desigual observado entre elas."""
    gb_pipe = Pipeline([("scaler", StandardScaler()),
                        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    gb_params = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    }
    skf_tuning = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_tuning = list(skf_tuning.split(X, question_ids))
    gs_gb = GridSearchCV(gb_pipe, gb_params, cv=cv_tuning, scoring="r2", n_jobs=-1)
    gs_gb.fit(X, y)
    print(f"   ↳ GradientBoosting tunado: melhores parâmetros = {gs_gb.best_params_}")
    return gs_gb.best_estimator_


def compute_metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "QWK": _qwk(y_true, y_pred),
        "Pearson": _safe_corr(pearsonr, y_true, y_pred),
        "Spearman": _safe_corr(spearmanr, y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# (A) Extração do dataset completo (uma única vez -- BERT é caro)
# --------------------------------------------------------------------------- #
def build_full_feature_matrix(data_path: str):
    print("📚 Carregando dados...")
    df = load_data(data_path)
    questions_map = load_questions()
    df = df.dropna(subset=["Base_answer", "Student_answer", "Grade"])
    print(f"✅ {len(df)} amostras carregadas")

    print("🤖 Carregando modelo de embeddings (uma única vez para todas as ablações)...")
    tokenizer, model = load_semantic_transformer()

    print("🔍 Extraindo features completas...")
    X, y, question_ids = [], [], []
    for idx, row in df.iterrows():
        try:
            student = str(row["Student_answer"])
            base = str(row["Base_answer"])
            grade = float(row["Grade"])
            qid = row["Question"]
            keywords = questions_map.get(qid, {}).get("keywords", [])

            from train_semantic_grader import get_bert_embeddings  # já importado acima também
            emb = get_bert_embeddings([student, base], tokenizer, model)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

            features = extract_features(student, base, sim, keywords)
            X.append(features)
            y.append(grade)
            question_ids.append(qid)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Erro linha {idx}: {e}")

    return np.array(X), np.array(y), np.array(question_ids)


# --------------------------------------------------------------------------- #
# (A) Ablação de features individuais
# --------------------------------------------------------------------------- #
def ablation_individual_features(X: np.ndarray, y: np.ndarray, question_ids: np.ndarray, kf):
    """Retorna (DataFrame de resultados, dict de hiperparâmetros tunados do
    GradientBoosting), para que `main()` repasse os mesmos hiperparâmetros
    a `ablation_hybrid_components` sem rodar o GridSearchCV de novo."""
    rows = []

    # Hiperparâmetros do GradientBoosting tunados UMA VEZ no conjunto completo
    # de features (mesma grade/protocolo de train_semantic_grader.py). Os
    # mesmos hiperparâmetros são então reaplicados (re-tunando o scaler, mas
    # não o GridSearchCV) a cada subconjunto com uma feature removida, para
    # isolar o efeito da feature em si, sem variar também a arquitetura do
    # modelo a cada rodada.
    print("🏋️ Tunando GradientBoosting (modelo completo, todas as features)...")
    best_gb_params = tuned_gb_pipeline(X, y, question_ids).named_steps["model"].get_params()
    gb_kwargs = {k: v for k, v in best_gb_params.items()
                 if k in ("n_estimators", "learning_rate", "max_depth")}

    def _eval(X_subset, label):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", GradientBoostingRegressor(
                             random_state=RANDOM_STATE, **gb_kwargs))])
        oof = cross_val_predict(pipe, X_subset, y, cv=kf)
        m = compute_metrics(y, oof)
        rows.append({"Configuração": label, "N_features": X_subset.shape[1], **m})

    # modelo completo (todas as features) -- baseline
    _eval(X, "Completo (todas as features)")

    # remove uma feature por vez
    for i, fname in enumerate(FEATURE_NAMES):
        mask = [j for j in range(X.shape[1]) if j != i]
        _eval(X[:, mask], f"Sem '{fname}'")

    full_mae = rows[0]["MAE"]
    full_qwk = rows[0]["QWK"]
    for r in rows[1:]:
        r["ΔMAE_vs_completo"] = r["MAE"] - full_mae  # positivo = piorou (MAE subiu)
        r["ΔQWK_vs_completo"] = r["QWK"] - full_qwk  # negativo = piorou (QWK caiu)
    rows[0]["ΔMAE_vs_completo"] = 0.0
    rows[0]["ΔQWK_vs_completo"] = 0.0

    cols = ["Configuração", "N_features", "MAE", "ΔMAE_vs_completo", "RMSE",
            "R2", "QWK", "ΔQWK_vs_completo", "Pearson", "Spearman"]
    return pd.DataFrame(rows)[cols], gb_kwargs


# --------------------------------------------------------------------------- #
# (B) Ablação dos componentes do score híbrido (60% regressor / 40% keywords)
# --------------------------------------------------------------------------- #
def ablation_hybrid_components(X: np.ndarray, y: np.ndarray, kf, gb_kwargs: dict,
                               semantic_weight: float = 0.6) -> pd.DataFrame:
    """
    Reproduz a composição híbrida (Seção III-D): nota_final = w*regressor + (1-w)*keywords.

    O "componente de regressão semântico-estrutural" é a predição out-of-fold de um
    GradientBoostingRegressor (tunado por GridSearchCV no mesmo protocolo de
    train_semantic_grader.py -- ver `gb_kwargs`) treinado com TODAS as features
    (incluindo similarity e keyword_ratio, como no sistema original -- ver Seção
    III-C do TCC: o keyword_ratio também entra como feature do regressor, além
    de compor a nota final diretamente). Já o "componente de cobertura de
    conteúdo" é o keyword_ratio bruto, usado diretamente como nota (0..1), sem
    passar pelo regressor.

    `gb_kwargs` deve conter os hiperparâmetros já tunados (n_estimators,
    learning_rate, max_depth) -- reaproveitados de ablation_individual_features
    para não rodar o GridSearchCV de novo.
    """
    rows = []

    # 1) Apenas o componente semântico-estrutural (GradientBoosting tunado, peso 100%)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", GradientBoostingRegressor(
                         random_state=RANDOM_STATE, **gb_kwargs))])
    oof_regressor = np.clip(cross_val_predict(pipe, X, y, cv=kf), 0, 1)
    rows.append({"Componente": "Apenas semântico-estrutural (GradientBoosting, peso 100%)",
                 **compute_metrics(y, oof_regressor)})

    # 2) Apenas cobertura de palavras-chave (keyword_ratio bruto como nota, peso 100%)
    keyword_score = np.clip(X[:, KEYWORD_IDX], 0, 1)
    rows.append({"Componente": "Apenas cobertura de palavras-chave (peso 100%)",
                 **compute_metrics(y, keyword_score)})

    # 3) Apenas similaridade BERT bruta (sem regressor, sem keywords) -- baseline adicional
    similarity_score = np.clip(X[:, SIMILARITY_IDX], 0, 1)
    rows.append({"Componente": "Apenas similaridade de cosseno (BERT bruto, sem regressor)",
                 **compute_metrics(y, similarity_score)})

    # 4) Híbrido completo, como no sistema em produção (60/40 por padrão)
    hybrid_score = np.clip(
        semantic_weight * oof_regressor + (1 - semantic_weight) * keyword_score, 0, 1
    )
    rows.append({"Componente": f"Híbrido completo "
                               f"({int(semantic_weight*100)}% semântico / "
                               f"{int((1-semantic_weight)*100)}% palavras-chave)",
                 **compute_metrics(y, hybrid_score)})

    cols = ["Componente", "MAE", "RMSE", "R2", "QWK", "Pearson", "Spearman"]
    return pd.DataFrame(rows)[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("Service", "data", "answers.json"),
                    help="caminho para o dataset (mesmo formato do train_semantic_grader.py)")
    ap.add_argument("--outdir", default=".", help="pasta de saída para os CSVs")
    ap.add_argument("--hybrid-weight", type=float, default=0.6,
                    help="peso do componente semântico no score híbrido (default 0.6, "
                         "conforme Seção III-D do TCC)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"❌ Arquivo não encontrado: {args.data}")
    os.makedirs(args.outdir, exist_ok=True)

    X, y, question_ids = build_full_feature_matrix(args.data)
    if len(X) < N_SPLITS:
        sys.exit(f"❌ Poucos dados para {N_SPLITS}-fold CV (apenas {len(X)} amostras).")

    # CV externa estratificada por questão (mesmo motivo documentado em
    # train_semantic_grader.py): garante composição comparável de cada
    # questão em todos os folds, evitando variância espúria nas métricas
    # de ablação dado o desempenho desigual observado entre questões.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    kf = list(skf.split(X, question_ids))

    print("\n" + "=" * 78)
    print("(A) ABLAÇÃO — FEATURES ESTRUTURAIS INDIVIDUAIS")
    print(f"    Validação cruzada {N_SPLITS}-fold estratificada por questão, "
          f"predições out-of-fold")
    print("=" * 78)
    feat_df, gb_kwargs = ablation_individual_features(X, y, question_ids, kf)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(feat_df.to_string(index=False))
    p1 = os.path.join(args.outdir, "ablation_features_individuais.csv")
    feat_df.to_csv(p1, index=False)
    print(f"\n💾 {p1}")

    print("\n" + "=" * 78)
    print("(B) ABLAÇÃO — COMPONENTES DO SCORE HÍBRIDO")
    print("=" * 78)
    hybrid_df = ablation_hybrid_components(X, y, kf, gb_kwargs,
                                           semantic_weight=args.hybrid_weight)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(hybrid_df.to_string(index=False))
    p2 = os.path.join(args.outdir, "ablation_componentes_hibrido.csv")
    hybrid_df.to_csv(p2, index=False)
    print(f"\n💾 {p2}")

    print("\n" + "=" * 78)
    print("LEITURA RÁPIDA")
    print("=" * 78)
    worst_row = feat_df.iloc[1:].loc[feat_df.iloc[1:]["ΔMAE_vs_completo"].idxmax()]
    print(f"-> Feature mais importante (maior aumento de MAE ao remover): "
          f"{worst_row['Configuração']} (ΔMAE={worst_row['ΔMAE_vs_completo']:+.4f})")
    best_neg = feat_df.iloc[1:].loc[feat_df.iloc[1:]["ΔMAE_vs_completo"].idxmin()]
    if best_neg["ΔMAE_vs_completo"] < 0:
        print(f"-> Possível feature redundante/ruidosa (MAE MELHOROU ao remover): "
              f"{best_neg['Configuração']} (ΔMAE={best_neg['ΔMAE_vs_completo']:+.4f})")
    best_component = hybrid_df.loc[hybrid_df["MAE"].idxmin()]
    print(f"-> Melhor componente isolado/combinação por MAE: "
          f"{best_component['Componente']} (MAE={best_component['MAE']:.4f})")


if __name__ == "__main__":
    main()
