import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = pd.read_csv("Datos.csv")

X = data.drop('Tiene novia', axis=1)
y = data['Tiene novia']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print(y_pred)

tree.export_graphviz(classifier, out_file="tree.dot",filled = True)


