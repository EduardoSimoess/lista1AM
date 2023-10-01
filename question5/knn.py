import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df_train = pd.read_excel('training_dataset.xlsx')

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

df_classify = pd.read_excel('test_dataset.xlsx')

X_new = df_classify.iloc[:, :-1]

X_new = scaler.transform(X_new)

predictions = model.predict(X_new)

print(predictions)

#Os resultados obtidos foram os mesmos em ambas as abordagens