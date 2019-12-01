# Polynomial Regression: Office Prices
# Link: https://www.hackerrank.com/challenges/predicting-office-space-price/problem
# Developer: Keith Leung

# Enter your code here. Read input from STDIN. Print output to STDOUT
# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# F = number of features, N = number of rows
F, N = map(int, input().split())


X_full, X_train, Y, X_test = [], [], [], []

# Creating X_full
for i in range(N):
    elements = list(map(float, input().split()))
    X_full.append(elements)
    
# Creating Y
for i in range(N):
    Y.append(X_full[i][F])

# Creating X_train
for i in range(N):
    x = []
    for j in range(F):
        if j < F:
            x.append(X_full[i][j])
    X_train.append(x)

T = int(input())

# Creating X_test
for i in range(T):
    elements = list(map(float, input().split()))
    X_test.append(elements)

# Set Polynomial Features
poly = PolynomialFeatures(degree=3)

# Set the model LinearRegression
model = LinearRegression()
model.fit(poly.fit_transform(X_train), Y)

prediction = model.predict(poly.fit_transform(X_test))
for i in range(len(prediction)):
    print(round(prediction[i],2))
