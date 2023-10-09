import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

df = pd.read_excel('dataset.xlsx')

df_encoded = pd.get_dummies(df, columns=['Crédito', 'Dívida', 'Garantia', 'Renda'])

X = df_encoded.drop('Risco', axis=1)
y = df_encoded['Risco']

id3_classifier = DecisionTreeClassifier(criterion='entropy')
id3_classifier.fit(X, y)

class_names = ['Alto', 'Moderado', 'Baixo']

dot_data = export_graphviz(id3_classifier, out_file=None, 
                           feature_names=X.columns,
                           class_names=class_names,
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)

graph.render("arvore_decision", view=True)  

