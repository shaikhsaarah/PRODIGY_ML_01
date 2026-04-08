!pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (mobile friendly)
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "bathrooms": [1, 2, 2, 3, 3],
    "price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)
df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
# Example prediction
pred = model.predict([[1800, 3, 2]])
print("Predicted Price:", pred[0])# User input (mobile-friendly)
area = 1800
bedrooms = 3
bathrooms = 2

prediction = model.predict([[area, bedrooms, bathrooms]])

print("Estimated House Price:", prediction[0])
from IPython.display import display

area = 1800  # change manually
bedrooms = 3
bathrooms = 2

prediction = model.predict([[area, bedrooms, bathrooms]])

print(f"📐 Area: {area}")
print(f"🛏 Bedrooms: {bedrooms}")
print(f"🚿 Bathrooms: {bathrooms}")
print(f"💰 Predicted Price: {prediction[0]}")
sns.pairplot(df)
plt.show()
