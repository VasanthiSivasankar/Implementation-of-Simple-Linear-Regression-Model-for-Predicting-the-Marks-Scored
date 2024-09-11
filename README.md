# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vasanthi Sivasankar
RegisterNumber:  212223040234
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print("dataset.head()")
print(dataset.head())
print("dataset.tail()")
print(dataset.tail())
print("dataset.info()")
dataset.info()
#assigning hours to X & scores to Y
print("X & Y values")
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
print("Prediction values of X & Y")
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,reg.predict(X_train),color="brown")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:

![1](https://github.com/user-attachments/assets/3a0cc4d3-7453-4207-85cb-faf691d29227)
![2](https://github.com/user-attachments/assets/9cedfbc6-d38e-411b-b185-150f305c8252)
![3](https://github.com/user-attachments/assets/a15bda70-8063-4de2-b3d4-c9fca79b7fbd)
![4](https://github.com/user-attachments/assets/4b105bc5-f7f2-4eef-a459-855845e51ea9)
![5](https://github.com/user-attachments/assets/13827152-f95c-41e4-82e6-c263c98e01c1)
![6](https://github.com/user-attachments/assets/d055a538-4def-4a33-82db-9a777f7de0f2)
![7](https://github.com/user-attachments/assets/5b9e3b52-0120-46c1-ace5-6bef58a0464d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
