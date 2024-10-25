#fonte do exemplo: https://www.datacamp.com/tutorial/decision-tree-classification-python

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from six import StringIO
from IPython.display import Image  
import pydotplus

# Configurar para mostrar todas as linhas e colunas
#pd.set_option('display.max_rows', None)  # Mostra todas as linhas
#pd.set_option('display.max_columns', None)  # Mostra todas as colunas

# fetch dataset 
# carregamento dos dados, baixado a base da internet inicialmente e agora carregada do arquivo local
df = pd.read_csv("data.csv")
col_names = df.columns.tolist()
print(col_names)
print(df.head())

# feature_cols = colunas de variaveis independentes, aquelas que conduzem ao resultado
feature_cols = col_names
del feature_cols[1]
print(feature_cols)
X = df[feature_cols] # Features
print(X)

# Target variable //variavel dependente, o resultado ao qual as demais colunas conduzem, aquilo que os algoritimos 
# precisam aprender a classificar
# cid é o nome da coluna na nossa base que significa cesoried identified: 
# 1 = falha, paciente morreu
# 0 = sucesso (tratamento com efeito)
y = df.cid 
print(y)

# Split dataset into training set and test set
# dividir duas vezes para atender ao trabalho (50% primeiro treino, e os outros 50% depois redividir em teste e validação)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

