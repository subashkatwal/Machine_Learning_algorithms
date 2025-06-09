import pandas as pd 

df = pd.read_csv("house_prices_dataset.csv")

# print(df.head())
# print(df.info())
# print(df.describe())


#Train the model 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X= df[['area']]
y= df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                        test_size= 0.2 , random_state=42)

model= LinearRegression()

model.fit(X_train, y_train)

print("Slope(m):", model.coef_[0])
print("Intercept (b):",model.intercept_)

#predict and evaluate 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 

y_pred= model.predict(X_test)

mae= mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)

rmse=np.sqrt(mse)

r2=r2_score(y_test,y_pred)

print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"RMSE:{rmse:.2f}")
print(f"R_Squared:{r2:.2f}")


#plot 

import matplotlib.pyplot as plt 

plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(), y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual VS predicted")
plt.show()


# Residual Plot (Homoscendasticity 