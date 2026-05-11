import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Carregar os dados
# Certifique-se que o arquivo está na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Conj_Resultados.csv')
df = pd.read_csv(file_path)

# 2. Função para converter texto (vírgula) em float
def limpar_nota(valor):
    if isinstance(valor, str):
        valor = valor.replace(',', '.').strip()
    try:
        return float(valor)
    except:
        return np.nan

# Limpar as colunas necessárias
colunas_notas = ['nota_original', 'nota_tcc', 'nota_gpt', 'nota_gemini']
for col in colunas_notas:
    df[col] = df[col].apply(limpar_nota)

# 3. FILTRO: Selecionar apenas notas originais entre 0.3 e 0.75
# O símbolo '&' significa 'e' (ambas as condições devem ser verdadeiras)
df_filtrado = df[(df['nota_original'] >= 0.5) & (df['nota_original'] <= 1.0) & (df['QID'] == 5)].dropna()

print(f"Total de linhas encontradas no intervalo [0.0 - 0.5]: {len(df_filtrado)}")

# 4. Cálculo do RMSE para os dados filtrados
colunas_teste = ['nota_tcc', 'nota_gpt', 'nota_gemini']
valores_rmse = []

for col in colunas_teste:
    rmse = np.sqrt(mean_squared_error(df_filtrado['nota_original'], df_filtrado[col]))
    valores_rmse.append(rmse)

# 5. Gerar o Gráfico
plt.figure(figsize=(10, 6))
cores = ['#3498db', '#e74c3c', '#2ecc71']
barras = plt.bar(colunas_teste, valores_rmse, color=cores)

plt.ylabel('RMSE (Menor é melhor)')
plt.title('RMSE - Range de 0.5 - 1.0 questão 5')

# Adicionar valores no topo das barras
for barra in barras:
    yval = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, yval + 0.02, 
             round(yval, 4), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Exibir os valores no console para conferência
for i, col in enumerate(colunas_teste):
    print(f"RMSE {col}: {valores_rmse[i]:.4f}")