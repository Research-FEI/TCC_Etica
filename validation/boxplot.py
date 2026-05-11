import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIGURAÇÕES
# =========================
QID_FILTRO = 5
NOTA_MIN = 0.0
NOTA_MAX = 1.0

COLUNAS_MODELOS = ['nota_tcc', 'nota_gpt', 'nota_gemini']

# =========================
# 1. CARREGAR DADOS
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Conj_Resultados.csv')

df = pd.read_csv(file_path)

# =========================
# 2. LIMPEZA DE DADOS
# =========================
def limpar_nota(valor):
    if isinstance(valor, str):
        valor = valor.replace(',', '.').strip()
    try:
        return float(valor)
    except:
        return np.nan

colunas_notas = ['nota_original'] + COLUNAS_MODELOS

for col in colunas_notas:
    df[col] = df[col].apply(limpar_nota)

# =========================
# 3. FILTRO
# =========================
df_filtrado = df[
    (df['nota_original'] >= NOTA_MIN) &
    (df['nota_original'] <= NOTA_MAX) 
    # (df['QID'] == QID_FILTRO)
].dropna()

print(f"\nTotal de amostras: {len(df_filtrado)}")

# =========================
# 4. MÉTRICAS
# =========================
resultados = []

for col in COLUNAS_MODELOS:
    rmse = np.sqrt(mean_squared_error(df_filtrado['nota_original'], df_filtrado[col]))
    mae = mean_absolute_error(df_filtrado['nota_original'], df_filtrado[col])

    resultados.append({
        'Modelo': col,
        'RMSE': rmse,
        'MAE': mae
    })

df_metricas = pd.DataFrame(resultados).sort_values(by='RMSE')

print("\n===== MÉTRICAS =====")
print(df_metricas)

# =========================
# 5. CÁLCULO DOS ERROS
# =========================
for col in COLUNAS_MODELOS:
    df_filtrado[f'erro_{col}'] = abs(df_filtrado[col] - df_filtrado['nota_original'])

# =========================
# 6. FORMATO LONGO (BOXPLOT)
# =========================
df_erros = pd.melt(
    df_filtrado,
    value_vars=[f'erro_{col}' for col in COLUNAS_MODELOS],
    var_name='modelo',
    value_name='erro'
)

df_erros['modelo'] = df_erros['modelo'].str.replace('erro_', '')

# Ordenar modelos pelo erro médio
ordem = df_erros.groupby('modelo')['erro'].mean().sort_values().index

# =========================
# 7. BOXPLOT
# =========================
plt.figure(figsize=(10, 6))

sns.boxplot(
    x='modelo',
    y='erro',
    data=df_erros,
    order=ordem
)

plt.title(f'Distribuição dos Erros (Q{QID_FILTRO} | {NOTA_MIN}-{NOTA_MAX})')
plt.ylabel('Erro Absoluto')
plt.xlabel('Modelo')

plt.tight_layout()
plt.show()

# =========================
# 8. SCATTER (REAL vs PREDITO)
# =========================
plt.figure(figsize=(8, 8))

for col in COLUNAS_MODELOS:
    plt.scatter(
        df_filtrado['nota_original'],
        df_filtrado[col],
        alpha=0.6,
        label=col
    )

# Linha ideal
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('Nota Real')
plt.ylabel('Nota Predita')
plt.title('Predição vs Real')
plt.legend()

plt.tight_layout()
plt.show()

# =========================
# 9. GRÁFICO DE MÉTRICAS
# =========================
plt.figure(figsize=(10, 6))

plt.bar(df_metricas['Modelo'], df_metricas['RMSE'])

for i, v in enumerate(df_metricas['RMSE']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.title('Comparação de RMSE')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()

# =========================
# 10. INSIGHTS AUTOMÁTICOS
# =========================
melhor_modelo = df_metricas.iloc[0]['Modelo']

print("\n===== INSIGHT AUTOMÁTICO =====")
print(f"Melhor modelo (menor RMSE): {melhor_modelo}")

print("\nErro médio por modelo:")
print(df_erros.groupby('modelo')['erro'].mean().sort_values())