import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 

data = pd.DataFrame({
    'Hour_Studied':[1,2,3,4,5,6,7,8,9,3,2],
    "Passed":      [0,0,0,0,1,1,1,0,1,1,0]
})
X= data[["Hour_Studied"]]
y=data["Passed"]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,
                                                   random_state=42)

model=LogisticRegression()

model.fit(X_train,y_train)

#predictions
y_prediction=model.predict(X_test)
probs=model.predict_proba(X_test)


print("Predictions:",y_prediction)
print("Probabilities:\n", probs)

print("Accuracy:", accuracy_score(y_test,y_prediction))
print("Confusion matrix:\n", confusion_matrix(y_test,y_prediction))


#plot the results 
import matplotlib.pyplot as plt 

z= np.linspace(-10,10,200)

def sigmoid(z):
    return 1/(1+np.exp(-z))

sig= sigmoid(z)

plt.figure(figsize=(8, 5))
plt.plot(z, sig, label='Sigmoid Function', color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='red', linestyle='--', label='z = 0 (P = 0.5)')
plt.title("Sigmoid Function in Logistic Regression")
plt.xlabel("z = wx + b")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.legend()
plt.show()