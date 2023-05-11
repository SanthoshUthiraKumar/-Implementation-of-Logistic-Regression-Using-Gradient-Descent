# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary library.
 
2. Load the text file in the compiler.
 
3. Plot the graphs using sigmoid , costfunction and gradient descent.

4. Predict the values.

5. End the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Santhosh U
RegisterNumber:  212222240092
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1 (2).txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def signoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,signoid(x_plot))
plt.show()

def costFunction(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return J,grad
  
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

def cost(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return J

def gradient(theta,x,y):
  h=signoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 Score")
  plt.ylabel("Exam 2 Score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=signoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=signoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
  
np.mean(predict(res.x,x)==y)

```

## Output:
### 1. Array Value of x
![Output1](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/d399b30b-92fc-48d7-b02b-0890c75a184e)

### 2. Array Value of y
![Output2](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/477020d8-74ef-4bd5-bf1d-8db3ffebb1d0)

### 3. Exam 1 - score graph
![Output3](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/a9ed9059-f989-4baf-a2b9-1bbfc4592a9e)

### 4. Sigmoid function graph
![Output4](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/4ed91b50-b30d-43d7-bd9d-8704dda7f0de)

### 5. X_train_grad value
![Output5](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/d403d1fa-97ce-4815-8f66-c8b8912728cf)

### 6. Y_train_grad value
![Output6](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/cbc7e614-d9f4-433b-8032-7034f2c759f8)

### 7. Print res.x
![Output7](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/99b2810f-f66b-418a-9d8c-b6ec542dae20)

### 8. Decision boundary - graph for exam score
![Output8](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/97643013-b2ef-4973-8573-5bb397544336)

### 9. Proability value 
![Output9](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/f42a1636-2088-4300-95b9-f93aaa60530c)

### 10. Prediction value of mean
![Output10](https://github.com/SanthoshUthiraKumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477975/aba94b66-db70-482d-9baa-9b5bead3c10e)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

