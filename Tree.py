import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree

nombre_col = ['Sexo','Tiene novia']

data = pd.read_csv('Datos.csv',header =None,names=nombre_col)

X = data.drop('Tiene novia', axis=1)
y = data['Tiene novia']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

tree.export_graphviz(classifier, out_file="tree.dot",filled = True,class_names = nombre_col)

pred = classifier.predict(X_test)
print(pred)
print("Accuracy:",metrics.accuracy_score(y_test, pred))
