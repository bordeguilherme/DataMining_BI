import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Importação e Visualização de Dados
# Carregar o dataset Wine da biblioteca sklearn e transformá-lo em um DataFrame do pandas para análise
data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)

# Exibir as primeiras linhas do dataset para uma visão inicial dos dados
print(wine_df.head())
print("\nDescrição das variáveis:")
# Exibir uma breve explicação das variáveis com base na documentação do dataset
for i, feature in enumerate(data.feature_names):
    print(f"{feature}: {data['DESCR'].split('\n')[i+16]}") 

# 2. Aplicação do Algoritmo K-means
# Escalar os dados para padronizá-los, melhorando a eficiência do algoritmo K-means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(wine_df)

# Configurar o algoritmo K-means para formar 3 clusters, pois sabemos que o dataset contém 3 classes de uvas
kmeans = KMeans(n_clusters=3, random_state=42)
# Atribuir a cada vinho o cluster correspondente e adicionar essa informação ao DataFrame
wine_df['Cluster'] = kmeans.fit_predict(scaled_data)

# Exibir as primeiras linhas do DataFrame com a nova coluna de cluster
print("\nClusters atribuídos:\n", wine_df[['Cluster']].head())

# 3. Análise de Correlação
# Calcular a matriz de correlação para identificar como as variáveis do dataset se relacionam entre si
correlation_matrix = wine_df.drop('Cluster', axis=1).corr()

# Exibir a matriz de correlação como um mapa de calor para facilitar a visualização das relações entre as variáveis
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

# Identificar e listar pares de variáveis com correlação moderada (entre 0,3 e 0,7)
correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if 0.3 <= abs(correlation_matrix.iloc[i, j]) <= 0.7:
            correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

print("\nPares de variáveis com correlação moderada (0.3 <= |correlação| <= 0.7):")
for pair in correlated_pairs:
    print(f"{pair[0]} e {pair[1]} - Correlação: {pair[2]:.2f}")

# 4. Visualização dos Clusters
# Selecionar duas variáveis com correlação moderada para criar um gráfico de dispersão dos clusters
# Exemplo: 'alcohol' e 'ash' (substituir conforme os pares encontrados na análise de correlação)
variable1 = correlated_pairs[0][0]  # Escolha de variáveis com base nos pares identificados
variable2 = correlated_pairs[0][1]

# Plotar um gráfico de dispersão para visualizar a distribuição dos clusters usando as variáveis selecionadas
plt.figure(figsize=(10, 6))
sns.scatterplot(data=wine_df, x=variable1, y=variable2, hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title(f"Distribuição dos Clusters com {variable1} vs {variable2}")
plt.xlabel(variable1)
plt.ylabel(variable2)
plt.legend(title='Cluster')
plt.show()

# 5. Conclusão
# Resumo dos principais pontos observados na análise de cluster e na correlação entre variáveis
print("\nConclusão:")
print("A análise de clusters permitiu observar padrões entre os diferentes tipos de vinho com base em suas características químicas.")
print("As variáveis selecionadas, como", variable1, "e", variable2, "têm correlação moderada e ajudaram a diferenciar os clusters, indicando possíveis distinções químicas entre as classes de vinho.")
