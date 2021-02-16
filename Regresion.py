import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd  

nombre_col = ['Sexo','Tiene novia']

data = np.array(
    [[0,1],
    [0,1],
    [1,1],
    [0,0],
    [0,0],
    [0,0],
    [0,1],
    [1,0],
    [0,0],
    [0,1],
    [1,1],
    [1,0],
    [0,1],
    [0,0],
    [0,1],
    [0,0],
    [0,1],
    [1,0],
    [1,1],
    [0,0],
    [0,1]
    ]
)
np.reshape(data,(-1,1))

X = data[:, 0:1].astype(int)  
y = data[:, 1].astype(int)


from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0)  
regressor.fit(X, y) 

y_pred = regressor.predict(np.array(4).reshape(1,1)) 

print("Predicted price: % d\n"% y_pred)  

