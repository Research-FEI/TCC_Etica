import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
RESULTS_CSV = os.path.join('api', 'results.csv')
COMPARISON_CSV = os.path.join('validation', 'Conj_Resultados.csv')

def clean_nota(valor):
    if isinstance(valor, str):
        valor = valor.replace(',', '.').strip()
    try:
        return float(valor)
    except:
        return np.nan

def calculate_mape(y_true, y_pred):
    # Usamos epsilon para evitar divisão por zero
    epsilon = 0.01 
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def generate_report():
    print("="*70)
    print("       🚀 RELATÓRIO EXAUSTIVO DE DESEMPENHO: IA vs. HUMANO")
    print("="*70)

    # 1. Load Local Results
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Local results not found at {RESULTS_CSV}.")
        return
    df_local = pd.read_csv(RESULTS_CSV)
    df_local['original_grade'] = pd.to_numeric(df_local['original_grade'], errors='coerce')
    df_local['evaluated_grade'] = pd.to_numeric(df_local['evaluated_grade'], errors='coerce')
    df_local = df_local.dropna(subset=['original_grade', 'evaluated_grade'])

    # Core Metrics for Local
    mae_l = mean_absolute_error(df_local['original_grade'], df_local['evaluated_grade'])
    rmse_l = np.sqrt(mean_squared_error(df_local['original_grade'], df_local['evaluated_grade']))
    mape_l = calculate_mape(df_local['original_grade'], df_local['evaluated_grade'])
    corr_l = df_local['original_grade'].corr(df_local['evaluated_grade'])

    print(f"Total de Amostras: {len(df_local)}")
    print(f"✅ Erro Médio (MAE): {mae_l:.4f}")
    print(f"📏 Erro Quadrático (RMSE): {rmse_l:.4f} (Penaliza erros grandes)")
    print(f"📉 Erro Percentual (MAPE): {mape_l:.2f}%")
    print(f"📈 Correlação: {corr_l:.4f}")

    # 2. Load Global LLM Comparison (if available)
    if os.path.exists(COMPARISON_CSV):
        print("\n" + "-"*70)
        print("📊 COMPARATIVO ENTRE MODELOS (BENCHMARK)")
        print("-"*70)
        df_comp = pd.read_csv(COMPARISON_CSV)
        
        cols = ['nota_original', 'nota_tcc', 'nota_gpt', 'nota_gemini']
        for col in cols:
            if col in df_comp.columns:
                df_comp[col] = df_comp[col].apply(clean_nota)
        df_comp = df_comp.dropna(subset=cols)

        metrics = []
        for model_col, label in [('nota_tcc', 'Nosso Modelo'), 
                                 ('nota_gpt', 'GPT-4o'), 
                                 ('nota_gemini', 'Gemini 1.5 Pro')]:
            mae = mean_absolute_error(df_comp['nota_original'], df_comp[model_col])
            rmse = np.sqrt(mean_squared_error(df_comp['nota_original'], df_comp[model_col]))
            mape = calculate_mape(df_comp['nota_original'], df_comp[model_col])
            corr = df_comp['nota_original'].corr(df_comp[model_col])
            metrics.append({
                'Modelo': label, 
                'MAE (↓)': round(mae, 3), 
                'RMSE (↓)': round(rmse, 3),
                'MAPE % (↓)': round(mape, 1),
                'Corr (↑)': round(corr, 3)
            })
        
        df_metrics = pd.DataFrame(metrics)
        print(df_metrics.to_string(index=False))

        # --- GRAPHICS ---
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(18, 5))

        # Plot 1: MAE vs RMSE
        plt.subplot(1, 3, 1)
        df_melt = df_metrics.melt(id_vars='Modelo', value_vars=['MAE (↓)', 'RMSE (↓)'], var_name='Métrica', value_name='Valor')
        sns.barplot(x='Modelo', y='Valor', hue='Métrica', data=df_melt)
        plt.title('Precisão Absoluta (MAE vs RMSE)')

        # Plot 2: MAPE
        plt.subplot(1, 3, 2)
        sns.barplot(x='Modelo', y='MAPE % (↓)', data=df_metrics, palette='viridis')
        plt.title('Erro Percentual Médio (MAPE)')

        # Plot 3: Correlation
        plt.subplot(1, 3, 3)
        sns.barplot(x='Modelo', y='Corr (↑)', data=df_metrics, palette='magma')
        plt.title('Sincronia com o Humano')

        plt.tight_layout()
        plt.savefig('comparativo_metricas_avancadas.png', dpi=300)
        print(f"\n✅ Novos gráficos salvos em: comparativo_metricas_avancadas.png")

    print("="*70)

if __name__ == "__main__":
    generate_report()
