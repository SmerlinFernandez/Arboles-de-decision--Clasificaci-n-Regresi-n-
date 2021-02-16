from inspect import classify_class_attrs
import numpy as np
import pandas as pd
from pandas.core.construction import array
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 

nombre_col = ['edad','tiene novia']

df = pd.read_csv('datosR.csv',names=nombre_col)

X = df.drop('tiene novia',axis=1)
y = df['tiene novia']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)

clasificador = DecisionTreeClassifier()

clasificador.fit(X_train,y_train)

y_pred = clasificador.predict(X_test)
print(y_pred)

precision = metrics.accuracy_score(y_test,y_pred)
print(precision)



tree.export_graphviz(clasificador, out_file="tree.dot",filled = True,class_names = nombre_col)