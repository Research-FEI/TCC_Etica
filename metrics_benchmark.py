#!/usr/bin/env python3
"""
metrics_benchmark.py
====================
Calcula as métricas de comparação para o benchmark de múltiplos avaliadores a partir
do CSV gerado por `benchmark_llm.py`:

    question_id, answer, professor_grade, model_tcc_grade, chatgpt_grade, gemini_grade

Para CADA modelo (Modelo TCC, ChatGPT, Gemini) contra a nota do professor, calcula:
  - MAE  (erro médio absoluto)
  - RMSE (penaliza erros grandes)
  - QWK  (Quadratic Weighted Kappa — concordância, notas binadas em 0..10)
  - Pearson  (correlação linear)
  - Spearman (correlação de postos / monotônica)

Gera:
  - tabela global por modelo  -> impressa + benchmark_global_metrics.csv
  - tabela por questão         -> impressa + benchmark_metrics_per_question.csv
  - gráficos (opcionais; use --no-plots para desligar)

Uso:
    python metrics_benchmark.py                          # usa benchmark_results.csv
    python metrics_benchmark.py --input outro.csv --outdir metrics/output
    python metrics_benchmark.py --no-plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

REFERENCE_COL = "professor_grade"

# rótulo amigável -> nome da coluna no CSV
MODEL_COLUMNS = {
    "Modelo TCC": "model_tcc_grade",
    "ChatGPT": "chatgpt_grade",
    "Gemini": "gemini_grade",
}


def _safe(fn, *args, default=np.nan):
    """Executa uma métrica protegendo contra entradas degeneradas (constantes, etc.)."""
    try:
        val = fn(*args)
        return float(val)
    except Exception:  # noqa: BLE001
        return default


def qwk(y_true, y_pred):
    """Quadratic Weighted Kappa com as notas 0..1 binadas em inteiros 0..10."""
    t = np.round(np.asarray(y_true) * 10).astype(int)
    p = np.round(np.asarray(y_pred) * 10).astype(int)
    return cohen_kappa_score(t, p, weights="quadratic", labels=list(range(11)))


def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def compute_global(df: pd.DataFrame, present_models: dict) -> pd.DataFrame:
    rows = []
    for label, col in present_models.items():
        pair = df[[REFERENCE_COL, col]].dropna()
        if len(pair) == 0:
            continue
        yt, yp = pair[REFERENCE_COL].to_numpy(), pair[col].to_numpy()
        rows.append({
            "Modelo": label,
            "N": len(pair),
            "MAE": _safe(mean_absolute_error, yt, yp),
            "RMSE": _safe(lambda a, b: np.sqrt(mean_squared_error(a, b)), yt, yp),
            "QWK": _safe(qwk, yt, yp),
            "Pearson": _safe(pearson, yt, yp),
            "Spearman": _safe(spearman, yt, yp),
            "Média": float(np.mean(yp)),
        })
    return pd.DataFrame(rows)


def compute_per_question(df: pd.DataFrame, present_models: dict) -> pd.DataFrame:
    rows = []
    for qid in sorted(df["question_id"].dropna().unique()):
        q_df = df[df["question_id"] == qid]
        for label, col in present_models.items():
            pair = q_df[[REFERENCE_COL, col]].dropna()
            if len(pair) == 0:
                continue
            yt, yp = pair[REFERENCE_COL].to_numpy(), pair[col].to_numpy()
            rows.append({
                "Questão": f"Q{int(qid)}",
                "Modelo": label,
                "N": len(pair),
                "MAE": _safe(mean_absolute_error, yt, yp),
                "RMSE": _safe(lambda a, b: np.sqrt(mean_squared_error(a, b)), yt, yp),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def make_plots(global_df, perq_df, outdir):
    import matplotlib
    matplotlib.use("Agg")  # backend sem display (servidor/sandbox)
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = {"Modelo TCC": "#2a9d8f", "ChatGPT": "#264653", "Gemini": "#e76f51"}

    # 1) Métricas de ERRO (menor = melhor)
    err = global_df.melt(id_vars="Modelo", value_vars=["MAE", "RMSE"],
                         var_name="Métrica", value_name="Valor")
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=err, x="Métrica", y="Valor", hue="Modelo", palette=palette)
    _annotate(ax)
    plt.title("Erro por modelo — menor é melhor (MAE e RMSE)", fontsize=14)
    plt.ylabel("Magnitude do erro (escala 0–1)")
    plt.ylim(0, max(err["Valor"].max() * 1.25, 0.1))
    plt.tight_layout()
    p1 = os.path.join(outdir, "benchmark_erro_por_modelo.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"✅ Gráfico salvo: {p1}")

    # 2) Métricas de CONCORDÂNCIA/CORRELAÇÃO (maior = melhor)
    agr = global_df.melt(id_vars="Modelo", value_vars=["QWK", "Pearson", "Spearman"],
                         var_name="Métrica", value_name="Valor")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agr, x="Métrica", y="Valor", hue="Modelo", palette=palette)
    _annotate(ax)
    plt.title("Concordância com o professor — maior é melhor "
              "(QWK, Pearson, Spearman)", fontsize=14)
    plt.ylabel("Coeficiente")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    p2 = os.path.join(outdir, "benchmark_concordancia_por_modelo.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"✅ Gráfico salvo: {p2}")

    # 3) MAE por questão, agrupado por modelo
    if not perq_df.empty:
        for metric, fname in (("MAE", "benchmark_mae_por_questao.png"),
                              ("RMSE", "benchmark_rmse_por_questao.png")):
            plt.figure(figsize=(11, 6))
            ax = sns.barplot(data=perq_df, x="Questão", y=metric, hue="Modelo",
                             palette=palette)
            _annotate(ax)
            plt.title(f"{metric} por questão e por modelo", fontsize=14)
            plt.ylabel("Magnitude do erro (escala 0–1)")
            plt.tight_layout()
            p = os.path.join(outdir, fname)
            plt.savefig(p, dpi=150); plt.close()
            print(f"✅ Gráfico salvo: {p}")


def _annotate(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and abs(h) > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom" if h >= 0 else "top",
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=9, fontweight="bold")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="benchmark_results.csv",
                    help="CSV gerado pelo benchmark_llm.py")
    ap.add_argument("--outdir", default=".",
                    help="pasta de saída para CSVs e gráficos")
    ap.add_argument("--no-plots", action="store_true", help="não gerar gráficos")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ Arquivo não encontrado: {args.input}\n"
                 f"   Rode antes: python benchmark_llm.py")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if REFERENCE_COL not in df.columns:
        sys.exit(f"❌ Coluna de referência '{REFERENCE_COL}' ausente no CSV.")

    # coage colunas numéricas
    for col in [REFERENCE_COL, *MODEL_COLUMNS.values()]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # mantém só modelos presentes e com pelo menos um valor
    present = {label: col for label, col in MODEL_COLUMNS.items()
               if col in df.columns and df[col].notna().any()}
    if not present:
        sys.exit("❌ Nenhuma coluna de modelo com notas válidas encontrada "
                 "(model_tcc_grade / chatgpt_grade / gemini_grade).")

    print(f"📊 Lendo {args.input} | {len(df)} linhas | "
          f"modelos: {', '.join(present)}\n")

    # ----- global -----
    global_df = compute_global(df, present)
    print("=" * 72)
    print("MÉTRICAS GLOBAIS (cada modelo vs. professor)")
    print("=" * 72)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(global_df.to_string(index=False))
    gpath = os.path.join(args.outdir, "benchmark_global_metrics.csv")
    global_df.to_csv(gpath, index=False)
    print(f"\n💾 {gpath}")

    # ----- por questão -----
    perq_df = compute_per_question(df, present)
    if not perq_df.empty:
        print("\n" + "=" * 72)
        print("MÉTRICAS POR QUESTÃO")
        print("=" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(perq_df.to_string(index=False))
        ppath = os.path.join(args.outdir, "benchmark_metrics_per_question.csv")
        perq_df.to_csv(ppath, index=False)
        print(f"\n💾 {ppath}")

    # ----- leitura rápida -----
    print("\n" + "=" * 72)
    print("LEITURA RÁPIDA")
    print("=" * 72)
    best_mae = global_df.loc[global_df["MAE"].idxmin()]
    print(f"-> Menor MAE: {best_mae['Modelo']} ({best_mae['MAE']:.4f})")
    if "QWK" in global_df:
        best_qwk = global_df.loc[global_df["QWK"].idxmax()]
        print(f"-> Maior concordância (QWK): {best_qwk['Modelo']} "
              f"({best_qwk['QWK']:.4f})")
    prof_mean = df[REFERENCE_COL].dropna().mean()
    print(f"-> Média do professor: {prof_mean:.2f}")
    for _, r in global_df.iterrows():
        tend = "infla" if r["Média"] > prof_mean else "subestima"
        print(f"   {r['Modelo']}: média {r['Média']:.2f} ({tend} em relação ao professor)")

    # ----- gráficos -----
    if not args.no_plots:
        try:
            make_plots(global_df, perq_df, args.outdir)
        except ImportError:
            print("\n⚠️ matplotlib/seaborn não instalados — pulei os gráficos. "
                  "Instale com: pip install matplotlib seaborn")
        except Exception as e:  # noqa: BLE001
            print(f"\n⚠️ Não foi possível gerar gráficos: {e}")


if __name__ == "__main__":
    main()
