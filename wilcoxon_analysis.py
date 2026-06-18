#!/usr/bin/env python3
"""
wilcoxon_analysis.py
=====================
Testes de Wilcoxon signed-rank (pareados, não-paramétricos) sobre o CSV gerado
por `benchmark_llm.py` (benchmark_results.csv):

    question_id, answer, professor_grade, model_tcc_grade, chatgpt_grade, gemini_grade

Executa DOIS tipos de comparação pareada:

  (A) ERRO ABSOLUTO pareado entre modelos
      |modelo_i - professor| vs |modelo_j - professor|, para cada par de modelos
      (Modelo TCC, ChatGPT, Gemini). Responde: "o erro de um modelo é
      sistematicamente menor/maior que o do outro, resposta a resposta?"

  (B) NOTA PREDITA vs NOTA DO PROFESSOR, por modelo
      modelo - professor, para cada modelo isoladamente. Responde: "o modelo
      difere sistematicamente do professor (infla ou subestima), de forma
      consistente o suficiente para não ser atribuível ao acaso?"

Em ambos os casos, o teste de Wilcoxon é apropriado porque as notas (0..1) não
seguem distribuição normal e a comparação é pareada (mesma resposta de aluno
avaliada por dois "juízes" diferentes).

Reporta também o tamanho de efeito (matched-pairs rank-biserial correlation, r),
porque p-valor sozinho não diz o quão grande é a diferença -- e os pares com
diferença nula (empates exatos) são reportados separadamente, já que o Wilcoxon
clássico os descarta (e isso pode importar muito quando os modelos truncam em
0,00 ou 1,00 com frequência).

Uso:
    python wilcoxon_analysis.py
    python wilcoxon_analysis.py --input outro_benchmark.csv --outdir resultados
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REFERENCE_COL = "professor_grade"

MODEL_COLUMNS = {
    "Modelo TCC": "model_tcc_grade",
    "ChatGPT": "chatgpt_grade",
    "Gemini": "gemini_grade",
}


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    """
    Tamanho de efeito matched-pairs rank-biserial (r), consistente com o
    teste de Wilcoxon signed-rank. r varia de -1 a 1; |r| pequeno ~0.1,
    médio ~0.3, grande ~0.5 (convenção usual em testes não-paramétricos).
    Pares com diferença zero são excluídos, como no próprio teste de Wilcoxon.
    """
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return float("nan")
    ranks = pd.Series(np.abs(diffs)).rank().to_numpy()
    pos_sum = ranks[diffs > 0].sum()
    neg_sum = ranks[diffs < 0].sum()
    total = pos_sum + neg_sum
    if total == 0:
        return float("nan")
    return float((pos_sum - neg_sum) / total)


def run_wilcoxon(a: np.ndarray, b: np.ndarray, label: str):
    """
    Roda Wilcoxon signed-rank entre dois vetores pareados (a vs b).
    Retorna um dict com estatística, p-valor, tamanho de efeito, N efetivo
    (excluindo empates) e quantos empates (zero_method='wilcox' descarta
    diferenças exatamente zero -- reportamos isso explicitamente).
    """
    diffs = a - b
    n_total = len(diffs)
    n_zero = int(np.sum(diffs == 0))
    n_effective = n_total - n_zero

    result = {
        "Comparação": label,
        "N_total": n_total,
        "N_empates_zero": n_zero,
        "N_efetivo": n_effective,
        "Diferença_média": float(np.mean(diffs)),
        "Diferença_mediana": float(np.median(diffs)),
    }

    if n_effective < 1:
        result.update({"Estatística_W": float("nan"), "p_valor": float("nan"),
                       "Efeito_r": float("nan"), "Significativo_(p<0.05)": "N/A (sem variação)"})
        return result

    try:
        # zero_method='wilcox' descarta pares empatados (comportamento padrão clássico);
        # method='auto' usa distribuição exata para N pequeno e normal aproximada para N grande
        stat, p = wilcoxon(a, b, zero_method="wilcox", method="auto")
    except ValueError as e:
        # ocorre se, após descartar empates, sobrar amostra degenerada
        result.update({"Estatística_W": float("nan"), "p_valor": float("nan"),
                       "Efeito_r": float("nan"),
                       "Significativo_(p<0.05)": f"N/A ({e})"})
        return result

    effect = rank_biserial_effect_size(diffs)
    result.update({
        "Estatística_W": float(stat),
        "p_valor": float(p),
        "Efeito_r": effect,
        "Significativo_(p<0.05)": "Sim" if p < 0.05 else "Não",
    })
    return result


def compare_model_errors(df: pd.DataFrame, present_models: dict) -> pd.DataFrame:
    """(A) Erro absoluto pareado entre cada par de modelos."""
    abs_err = {}
    for label, col in present_models.items():
        pair = df[[REFERENCE_COL, col]].dropna()
        abs_err[label] = pair.assign(
            _err=lambda d: (d[col] - d[REFERENCE_COL]).abs()
        )[["_err"]].rename(columns={"_err": label})

    labels = list(present_models.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            merged = abs_err[li].join(abs_err[lj], how="inner", lsuffix="_i", rsuffix="_j")
            # join por índice exige mesmo índice original (mesma linha do CSV);
            # como ambos vêm do df original sem reset, os índices já alinham.
            col_i = merged.columns[0]
            col_j = merged.columns[1]
            a = merged[col_i].to_numpy()
            b = merged[col_j].to_numpy()
            if len(a) == 0:
                continue
            rows.append(run_wilcoxon(a, b, f"|erro {li}| vs |erro {lj}|"))
    return pd.DataFrame(rows)


def compare_model_vs_professor(df: pd.DataFrame, present_models: dict) -> pd.DataFrame:
    """(B) Nota do modelo vs nota do professor, por modelo."""
    rows = []
    for label, col in present_models.items():
        pair = df[[REFERENCE_COL, col]].dropna()
        if len(pair) == 0:
            continue
        a = pair[col].to_numpy()
        b = pair[REFERENCE_COL].to_numpy()
        rows.append(run_wilcoxon(a, b, f"{label} vs Professor"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="benchmark_results.csv",
                    help="CSV gerado pelo benchmark_llm.py")
    ap.add_argument("--outdir", default=".",
                    help="pasta de saída para os CSVs de resultado")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ Arquivo não encontrado: {args.input}\n"
                 f"   Rode antes: python benchmark_llm.py")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if REFERENCE_COL not in df.columns:
        sys.exit(f"❌ Coluna de referência '{REFERENCE_COL}' ausente no CSV.")

    for col in [REFERENCE_COL, *MODEL_COLUMNS.values()]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = {label: col for label, col in MODEL_COLUMNS.items()
               if col in df.columns and df[col].notna().any()}
    if not present:
        sys.exit("❌ Nenhuma coluna de modelo com notas válidas encontrada.")

    print(f"📊 Lendo {args.input} | {len(df)} linhas | modelos: {', '.join(present)}\n")

    print("=" * 78)
    print("(A) WILCOXON — ERRO ABSOLUTO PAREADO ENTRE MODELOS")
    print("    H0: a distribuição das diferenças de erro absoluto é simétrica em torno de 0")
    print("    (nenhum dos dois modelos erra sistematicamente mais que o outro)")
    print("=" * 78)
    err_df = compare_model_errors(df, present)
    if not err_df.empty:
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(err_df.to_string(index=False))
        p1 = os.path.join(args.outdir, "wilcoxon_erro_pareado.csv")
        err_df.to_csv(p1, index=False)
        print(f"\n💾 {p1}")
    else:
        print("(menos de dois modelos disponíveis para comparar)")

    print("\n" + "=" * 78)
    print("(B) WILCOXON — NOTA DO MODELO vs NOTA DO PROFESSOR")
    print("    H0: a distribuição das diferenças (modelo - professor) é simétrica em torno de 0")
    print("    (o modelo não infla nem subestima sistematicamente)")
    print("=" * 78)
    prof_df = compare_model_vs_professor(df, present)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(prof_df.to_string(index=False))
    p2 = os.path.join(args.outdir, "wilcoxon_vs_professor.csv")
    prof_df.to_csv(p2, index=False)
    print(f"\n💾 {p2}")

    print("\n" + "=" * 78)
    print("LEITURA RÁPIDA")
    print("=" * 78)
    print("Nota: 'Efeito_r' é a correlação rank-biserial pareada; |r| pequeno~0.1, "
          "médio~0.3, grande~0.5.")
    print("Nota: empates exatos (diferença = 0) são excluídos do teste, conforme "
          "convenção do Wilcoxon signed-rank -- reportados em 'N_empates_zero'.")
    for _, r in prof_df.iterrows():
        if r["Significativo_(p<0.05)"] == "Sim":
            direcao = "infla" if r["Diferença_média"] > 0 else "subestima"
            print(f"  -> {r['Comparação']}: diferença significativa (p={r['p_valor']:.4g}), "
                  f"{direcao} sistematicamente em relação ao professor "
                  f"(Δmédia={r['Diferença_média']:.3f}, r={r['Efeito_r']:.3f})")
        else:
            print(f"  -> {r['Comparação']}: sem diferença sistemática significativa "
                  f"(p={r['p_valor']:.4g})")


if __name__ == "__main__":
    main()
