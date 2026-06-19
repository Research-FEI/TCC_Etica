#!/usr/bin/env python3
"""
feature_correlation.py
=======================
Gera a matriz de correlação entre as features estruturais usadas no regressor
(ver Seção III-C do TCC) e a nota do professor, como apoio à discussão de
multicolinearidade e à interpretação do ablation study.

Reaproveita a MESMA extração de features e embeddings do train_semantic_grader.py
(mesmo modelo de sentence-embeddings, mesmo dataset), para que a matriz reflita
exatamente os dados usados no treino e na ablação -- nenhuma extração é refeita
de forma divergente.

Gera:
  - matriz de correlação (Pearson e Spearman) entre as 7 features + nota do
    professor -> CSV
  - heatmap anotado (Pearson) -> PNG
  - heatmap anotado (Spearman) -> PNG
  - aviso de pares com |r| > 0.8 (candidatos a multicolinearidade)

IMPORTANTE: rode a partir da pasta api/ (mesmo requisito do ablation_study.py),
pois importa diretamente de train_semantic_grader.py.

Uso (a partir de api/):
    python feature_correlation.py
    python feature_correlation.py --data Service/data/answers.json --outdir ../resultados_features
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train_semantic_grader import (
        load_data, load_questions, load_semantic_transformer, extract_features,
        get_bert_embeddings, FEATURE_NAMES,
    )
except ImportError:
    sys.exit(
        "❌ Não foi possível importar train_semantic_grader.py.\n"
        "   Rode este script a partir da mesma pasta (api/), ou ajuste o sys.path."
    )

from sklearn.metrics.pairwise import cosine_similarity

GRADE_COL = "Nota_Professor"
MULTICOLLINEARITY_THRESHOLD = 0.8


# --------------------------------------------------------------------------- #
# Extração (mesma lógica do ablation_study.py / train_semantic_grader.py)
# --------------------------------------------------------------------------- #
def build_feature_dataframe(data_path: str) -> pd.DataFrame:
    print("📚 Carregando dados...")
    df = load_data(data_path)
    questions_map = load_questions()
    df = df.dropna(subset=["Base_answer", "Student_answer", "Grade"])
    print(f"✅ {len(df)} amostras carregadas")

    print("🤖 Carregando modelo de embeddings...")
    tokenizer, model = load_semantic_transformer()

    print("🔍 Extraindo features...")
    rows = []
    for idx, row in df.iterrows():
        try:
            student = str(row["Student_answer"])
            base = str(row["Base_answer"])
            grade = float(row["Grade"])
            qid = row["Question"]
            keywords = questions_map.get(qid, {}).get("keywords", [])

            emb = get_bert_embeddings([student, base], tokenizer, model)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

            features = extract_features(student, base, sim, keywords)
            record = dict(zip(FEATURE_NAMES, features))
            record[GRADE_COL] = grade
            rows.append(record)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Erro linha {idx}: {e}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Correlação e heatmaps
# --------------------------------------------------------------------------- #
def compute_correlations(feat_df: pd.DataFrame):
    pearson_corr = feat_df.corr(method="pearson")
    spearman_corr = feat_df.corr(method="spearman")
    return pearson_corr, spearman_corr


def flag_multicollinearity(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lista pares de FEATURES (excluindo a nota do professor) com |r| acima do limiar."""
    cols = [c for c in corr.columns if c != GRADE_COL]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append({"Feature_A": a, "Feature_B": b, "r": float(r)})
    return pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False) if pairs else pd.DataFrame(
        columns=["Feature_A", "Feature_B", "r"])


def make_heatmap(corr: pd.DataFrame, title: str, outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    plt.figure(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # mantém diagonal, esconde triângulo superior
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Coeficiente de correlação", "shrink": 0.8},
        annot_kws={"fontsize": 9},
    )
    plt.title(title, fontsize=13, pad=12)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"✅ Heatmap salvo: {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("Service", "data", "answers.json"),
                    help="caminho para o dataset (mesmo formato do train_semantic_grader.py)")
    ap.add_argument("--outdir", default=".", help="pasta de saída para CSVs e PNGs")
    ap.add_argument("--threshold", type=float, default=MULTICOLLINEARITY_THRESHOLD,
                    help="limiar de |r| para sinalizar multicolinearidade (default 0.8)")
    ap.add_argument("--no-plots", action="store_true", help="não gerar os heatmaps (só CSVs)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"❌ Arquivo não encontrado: {args.data}")
    os.makedirs(args.outdir, exist_ok=True)

    feat_df = build_feature_dataframe(args.data)
    if feat_df.empty:
        sys.exit("❌ Nenhuma amostra processada com sucesso.")

    # ordena colunas: features na ordem definida, nota do professor por último
    ordered_cols = [c for c in FEATURE_NAMES if c in feat_df.columns] + [GRADE_COL]
    feat_df = feat_df[ordered_cols]

    pearson_corr, spearman_corr = compute_correlations(feat_df)

    print("\n" + "=" * 78)
    print("MATRIZ DE CORRELAÇÃO — PEARSON (linear)")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(pearson_corr.to_string())
    p1 = os.path.join(args.outdir, "feature_correlation_pearson.csv")
    pearson_corr.to_csv(p1)
    print(f"\n💾 {p1}")

    print("\n" + "=" * 78)
    print("MATRIZ DE CORRELAÇÃO — SPEARMAN (postos/monotônica)")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(spearman_corr.to_string())
    p2 = os.path.join(args.outdir, "feature_correlation_spearman.csv")
    spearman_corr.to_csv(p2)
    print(f"\n💾 {p2}")

    print("\n" + "=" * 78)
    print(f"ALERTA DE MULTICOLINEARIDADE (|r| ≥ {args.threshold}, Pearson, só entre features)")
    print("=" * 78)
    flagged = flag_multicollinearity(pearson_corr, args.threshold)
    if flagged.empty:
        print("Nenhum par de features acima do limiar -- sem sinal forte de multicolinearidade.")
    else:
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(flagged.to_string(index=False))
        p3 = os.path.join(args.outdir, "feature_multicollinearity_flags.csv")
        flagged.to_csv(p3, index=False)
        print(f"\n💾 {p3}")

    print("\n" + "=" * 78)
    print("CORRELAÇÃO DE CADA FEATURE COM A NOTA DO PROFESSOR (Pearson, ordenado)")
    print("=" * 78)
    with_grade = pearson_corr[GRADE_COL].drop(GRADE_COL).sort_values(key=abs, ascending=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(with_grade.to_string())

    if not args.no_plots:
        try:
            make_heatmap(pearson_corr,
                         "Correlação de Pearson — Features Estruturais e Nota do Professor",
                         os.path.join(args.outdir, "heatmap_correlacao_pearson.png"))
            make_heatmap(spearman_corr,
                         "Correlação de Spearman — Features Estruturais e Nota do Professor",
                         os.path.join(args.outdir, "heatmap_correlacao_spearman.png"))
        except ImportError:
            print("\n⚠️ matplotlib/seaborn não instalados — pulei os heatmaps. "
                  "Instale com: pip install matplotlib seaborn")
        except Exception as e:  # noqa: BLE001
            print(f"\n⚠️ Não foi possível gerar os heatmaps: {e}")


if __name__ == "__main__":
    main()
