import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


data = {
    'Rice':        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    'Lentils':     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    'Vegetables':  [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'Meat':        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'Spices':      [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    'Fermented':   [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    'Dairy':       [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    'Sweet':       [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    'Sour':        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    'Hot_or_Cold': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    'Dish': ['Dal Bhat', 'Momo', 'Sel Roti', 'Gundruk', 'Yomari', 'Sukuti', 'Kheer', 'Aloo Tama', 'Choila',
             'Thukpa', 'Sekuwa', 'Rasbari', 'Barfi', 'Peda', 'Tama', 'Kinema', 'Khichadi', 'Dahi', 'Saag', 'Chowmein']
}
df = pd.DataFrame(data)
X = df.drop('Dish', axis=1)
y = df['Dish']
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# === Step 2: Tkinter UI ===
class FoodClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Nepali Food Classifier")
        self.master.geometry("400x300")
        self.features = ['Rice', 'Lentils', 'Vegetables', 'Meat', 'Spices',
                         'Fermented', 'Dairy', 'Sweet', 'Sour', 'Hot_or_Cold']
        self.answers = []
        self.q_index = 0

        self.label = tk.Label(master, text="Welcome to Nepali Food Classifier!", font=("Arial", 14))
        self.label.pack(pady=20)

        self.question = tk.Label(master, text="", font=("Arial", 12))
        self.question.pack(pady=10)

        self.yes_button = tk.Button(master, text="✅ Yes", command=lambda: self.record_answer(1), width=10)
        self.yes_button.pack(pady=5)

        self.no_button = tk.Button(master, text="❌ No", command=lambda: self.record_answer(0), width=10)
        self.no_button.pack(pady=5)

        self.next_question()

    def next_question(self):
        if self.q_index < len(self.features):
            current_feature = self.features[self.q_index]
            self.question.config(text=f"Does it contain {current_feature.replace('_', ' ')}?")
        else:
            self.predict_dish()

    def record_answer(self, ans):
        self.answers.append(ans)
        self.q_index += 1
        self.next_question()

    def predict_dish(self):
        user_input = [self.answers]
        prediction = model.predict(user_input)
        messagebox.showinfo("Prediction", f"🥘 Predicted Dish: {prediction[0]}")
        self.master.quit()


# === Step 3: Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FoodClassifierApp(root)
    root.mainloop()
