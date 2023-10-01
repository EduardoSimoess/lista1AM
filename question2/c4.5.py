import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Carregue seus dados do arquivo Excel
df = pd.read_excel('dataset.xlsx')

# Codifique colunas categóricas usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Crédito', 'Dívida', 'Garantia', 'Renda'])

# Separe os dados em recursos (X) e rótulos (y)
X = df_encoded.drop('Risco', axis=1)
y = df_encoded['Risco']

# Crie e treine o modelo de árvore de decisão com o algoritmo C4.5 usando todo o conjunto de dados
c45_classifier = DecisionTreeClassifier(criterion='entropy')
c45_classifier.fit(X, y)

# Defina as classes alvo
class_names = ['Alto', 'Moderado', 'Baixo']

# Exporte a árvore de decisão para um arquivo .dot
dot_data = export_graphviz(id3_classifier, out_file=None, 
                           feature_names=X.columns,
                           class_names=class_names,
                           filled=True, rounded=True, special_characters=True)

# Crie um objeto Graphviz a partir dos dados do .dot
graph = graphviz.Source(dot_data)

# Renderize a árvore em um arquivo PDF e visualize-o
graph.render("arvore_decision", view=True)  # Isso criará um arquivo "arvore_decision.pdf"
