
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def calculate_per_question_metrics(csv_path='api/results.csv'):
    if not os.path.exists(csv_path):
        print(f"❌ Erro: Arquivo {csv_path} não encontrado.")
        return

    print(f"📊 Analisando resultados por questão de: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['question_id', 'original_grade', 'evaluated_grade'])
        
        results = []
        
        # Analisa questões de 1 a 5
        for qid in range(1, 6):
            q_df = df[df['question_id'] == qid]
            
            if len(q_df) == 0:
                print(f"⚠️ Aviso: Nenhuma amostra encontrada para a questão {qid}")
                continue
                
            mae = mean_absolute_error(q_df['original_grade'], q_df['evaluated_grade'])
            rmse = np.sqrt(mean_squared_error(q_df['original_grade'], q_df['evaluated_grade']))
            
            results.append({
                'Questão': f'Q{qid}',
                'MAE': mae,
                'RMSE': rmse,
                'Amostras': len(q_df)
            })
            
            print(f"✅ Questão {qid}: MAE={mae:.4f}, RMSE={rmse:.4f} ({len(q_df)} amostras)")

        if not results:
            print("❌ Erro: Nenhum dado processado.")
            return

        # Criar DataFrame para plotagem
        res_df = pd.DataFrame(results)
        
        # Transformar para formato "long" (ideal para o Seaborn)
        plot_df = res_df.melt(id_vars='Questão', value_vars=['MAE', 'RMSE'], 
                              var_name='Métrica', value_name='Erro')

        # Gerar Gráfico
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 7))
        
        ax = sns.barplot(x='Questão', y='Erro', hue='Métrica', data=plot_df, palette=['skyblue', 'salmon'])
        
        # Adicionar valores no topo das barras
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontweight='bold')

        plt.title('Comparativo de Desempenho por Questão (MAE vs RMSE)', fontsize=16)
        plt.ylabel('Magnitude do Erro (Escala 0-1)', fontsize=12)
        plt.xlabel('ID da Questão', fontsize=12)
        plt.ylim(0, max(plot_df['Erro']) * 1.2)
        plt.legend(title='Métricas', loc='upper right')
        
        plt.tight_layout()
        plt.savefig('api/metrics_per_question.png')
        print("\n✅ Gráfico comparativo por questão salvo em: api/metrics_per_question.png")

    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")

if __name__ == "__main__":
    calculate_per_question_metrics()
