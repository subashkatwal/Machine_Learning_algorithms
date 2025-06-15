import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 

data = {
    'Rice':       [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    'Lentils':    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'Vegetables': [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    'Meat':       [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    'Spices':     [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    'Fermented':  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Dairy':      [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    'Sweet':      [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    'Sour':       [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    'Hot':        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],

    # New features:
    'Liked_by_Girls': [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    'Liked_by_Boys':  [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
    'Taken_Hot_or_Cold': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],

    'Dish': [
        'Dal Bhat', 'Momo', 'Sel Roti', 'Gundruk', 'Yomari',
        'Sukuti', 'Kheer', 'Aloo Tama', 'Choila', 'Thukpa',
        'Chatamari', 'Sekuwa', 'Fried Fish', 'Bara', 'Juju Dhau',
        'Kwati', 'Buff Curry', 'Vegetable Curry', 'Lassi', 'Sikarni',
        'Chatpate', 'Panipuri', 'Pani Chaat'
    ]
}

df = pd.DataFrame(data)

X = df.drop('Dish', axis=1)
y = df['Dish']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print("\nNepali Food Classifier")
print("\nThink of any food in your mind and answer the following questions:")
print("Please enter the following questions with 'yes' or 'no':\n")

features = list(X.columns)
user_input = []

for feature in features:
    ans = input(f"Does it contain or is it related to '{feature.replace('_', ' ')}'? (yes/no): ").strip().lower()
    user_input.append(1 if ans in ['yes', 'y'] else 0)

distances, indices = model.kneighbors([user_input])

print("\nPossible dishes you could be thinking of:")

for i in range(len(indices[0])):
    dish = y.iloc[indices[0][i]]
    dist = distances[0][i]
    print(f"{i+1}. {dish} (distance: {dist:.2f})")

prediction = model.predict(np.array(user_input).reshape(1, -1))
print("\nPredicted Nepali Dish:", prediction[0])
