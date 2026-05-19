
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix
import os

def generate_plots(df, correlation, mae):
    """Gera gráficos para a apresentação do TCC com títulos baseados nas métricas"""
    print("\n🎨 Gerando gráficos para a apresentação...")
    
    # Configuração de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]

    # Gráfico 1: Dispersão (Correção de Pearson)
    plt.figure()
    sns.regplot(x='original_grade', y='evaluated_grade', data=df, 
                scatter_kws={'alpha':0.5, 'color':'teal'}, 
                line_kws={'color':'red', 'label':f'Tendência (Corr: {correlation:.2f})'})
    plt.title(f'Métrica: Correlação de Pearson ({correlation:.2f})', fontsize=15)
    plt.xlabel('Nota do Professor (Humano)', fontsize=12)
    plt.ylabel('Nota da Inteligência Artificial', fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('api/grafico_dispersao.png')
    print("✅ Gráfico de dispersão salvo em: api/grafico_dispersao.png")

    # Gráfico 2: Comparação de Médias (Viés de Avaliação)
    plt.figure()
    mean_orig = df['original_grade'].mean()
    mean_eval = df['evaluated_grade'].mean()
    medias = [mean_orig, mean_eval]
    labels = [f'Humano (Média: {mean_orig:.2f})', f'IA (Média: {mean_eval:.2f})']
    sns.barplot(x=labels, y=medias, palette='viridis', hue=labels, legend=False)
    plt.title('Métrica: Comparação de Médias (Rigor de Avaliação)', fontsize=15)
    plt.ylabel('Média das Notas (0-1)', fontsize=12)
    plt.ylim(0, 1)
    # Adiciona os valores em cima das barras
    for i, v in enumerate(medias):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('api/comparativo_medias.png')
    print("✅ Gráfico de médias salvo em: api/comparativo_medias.png")

    # Gráfico 3: Distribuição do Erro Absoluto (MAE)
    plt.figure()
    df['erro_absoluto'] = (df['original_grade'] - df['evaluated_grade']).abs()
    sns.histplot(df['erro_absoluto'], kde=True, color='orange', bins=10)
    plt.axvline(mae, color='red', linestyle='--', label=f'MAE ({mae:.2f})')
    plt.title(f'Métrica: Distribuição do Erro Absoluto (MAE: {mae:.2f})', fontsize=15)
    plt.xlabel('Magnitude do Erro (em pontos)', fontsize=12)
    plt.ylabel('Frequência de Ocorrência', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('api/distribuicao_erros.png')
    print("✅ Gráfico de distribuição de erros salvo em: api/distribuicao_erros.png")

def generate_mae_rmse_comparison(mae, rmse):
    """Gera um gráfico comparando MAE e RMSE para destacar o impacto de outliers"""
    print("✅ Gerando gráfico comparativo MAE vs RMSE...")
    
    plt.figure(figsize=(8, 6))
    metrics = ['MAE (Erro Médio)', 'RMSE (Erro Penalizado)']
    values = [mae, rmse]
    
    # Criando o gráfico de barras
    colors = ['skyblue', 'salmon']
    bars = plt.bar(metrics, values, color=colors)
    
    # Adicionando os valores no topo das barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Diferença entre Erro Médio (MAE) e Erro com Outliers (RMSE)', fontsize=14)
    plt.ylabel('Magnitude do Erro (Escala 0-1)', fontsize=12)
    plt.ylim(0, max(values) * 1.3) # Espaço para o texto no topo
    
    # Adicionando uma anotação explicando a diferença
    diff = ((rmse - mae) / mae) * 100
    plt.annotate(f'RMSE é {diff:.1f}% maior que o MAE\n(Indica impacto de erros extremos)', 
                 xy=(1, rmse), xytext=(0.5, rmse + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('api/comparativo_mae_rmse.png')
    print("✅ Gráfico comparativo salvo em: api/comparativo_mae_rmse.png")

def generate_qwk_heatmap(df, qwk):
    """Gera uma matriz de confusão/calor para representar o QWK (Concordância)"""
    print("✅ Gerando mapa de calor de concordância (QWK)...")
    
    # Converter para escala 0-10 para a matriz
    y_true = np.round(df['original_grade'] * 10).astype(int)
    y_pred = np.round(df['evaluated_grade'] * 10).astype(int)
    
    # Criar a matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(11))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                xticklabels=range(11), yticklabels=range(11))
    
    plt.title(f'Matriz de Concordância (QWK: {qwk:.4f})\nFrequência de Notas: Humano vs IA', fontsize=14)
    plt.xlabel('Nota da IA (Escala 0-10)', fontsize=12)
    plt.ylabel('Nota do Humano (Escala 0-10)', fontsize=12)
    
    # Adicionar anotação sobre a diagonal
    # plt.text(5.5, -0.5, "Diagonal = Acerto Perfeito", ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('api/matriz_concordancia_qwk.png')
    print("✅ Matriz de concordância salva em: api/matriz_concordancia_qwk.png")

def calculate_project_metrics(csv_path='api/results.csv'):
    if not os.path.exists(csv_path):
        print(f"❌ Erro: Arquivo {csv_path} não encontrado.")
        return

    print(f"📊 Analisando resultados de: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Remove linhas com valores nulos
        df = df.dropna(subset=['original_grade', 'evaluated_grade'])
        
        if len(df) == 0:
            print("❌ Erro: O arquivo CSV está vazio ou sem notas válidas.")
            return

        # 1. MAE (Mean Absolute Error)
        mae = mean_absolute_error(df['original_grade'], df['evaluated_grade'])

        # 2. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(df['original_grade'], df['evaluated_grade']))

        # 3. QWK (Quadratic Weighted Kappa)
        orig_binned = np.round(df['original_grade'] * 10).astype(int)
        eval_binned = np.round(df['evaluated_grade'] * 10).astype(int)
        qwk = cohen_kappa_score(orig_binned, eval_binned, weights='quadratic')

        # 4. Correlação de Pearson
        correlation = df['original_grade'].corr(df['evaluated_grade'])

        print("-" * 30)
        print(f"Amostras analisadas: {len(df)}")
        print(f"MAE:   {mae:.4f}  (Erro médio absoluto)")
        print(f"RMSE:  {rmse:.4f}  (Penaliza erros grandes)")
        print(f"QWK:   {qwk:.4f}  (Concordância Humano-IA)")
        print(f"Corr:  {correlation:.4f}  (Tendência de acerto)")
        print("-" * 30)
        
        # Gera os gráficos
        generate_plots(df, correlation, mae)
        generate_mae_rmse_comparison(mae, rmse)
        generate_qwk_heatmap(df, qwk)

        # Insights Rápidos
        print("\n💡 Insights para a apresentação:")
        if rmse > mae * 1.2:
            print("-> O RMSE está bem acima do MAE: Isso confirma a presença de erros extremos (Outliers),")
            print("   provavelmente causados pelo filtro rígido de palavras-chave.")
        
        if correlation > 0.4:
            print("-> Correlação acima de 0.4: O modelo já consegue identificar a tendência de qualidade das respostas.")

    except Exception as e:
        print(f"❌ Ocorreu um erro ao processar: {e}")

if __name__ == "__main__":
    calculate_project_metrics()
