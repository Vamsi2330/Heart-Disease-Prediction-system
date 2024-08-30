import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare the data
heart_disease = pd.read_csv("E:\Heart-Disease-Prediction-system-main\Heart-Disease-Prediction-system-main\heart.csv")

X = heart_disease.drop(columns='target', axis=1)
Y = heart_disease['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Random Forest model
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=2)
classifier.fit(X_train, Y_train)


# Function to make predictions
def predict():
    try:
        input_data = [int(entry_age.get()), int(entry_sex.get()), int(entry_cp.get()), int(entry_trestbps.get()),
                      int(entry_chol.get()), int(entry_fbs.get()), int(entry_restecg.get()), int(entry_thalach.get()),
                      int(entry_exang.get()), float(entry_oldpeak.get()), int(entry_slope.get()), int(entry_ca.get()),
                      int(entry_thal.get())]

        input_data_as_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_array.reshape(1, -1)
        prediction = classifier.predict(input_data_reshaped)

        if prediction == 0:
            messagebox.showinfo("Prediction Result", "The person's heart is Healthy.")
        else:
            messagebox.showinfo("Prediction Result", "The person has heart disease.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all fields.")


# Set up the GUI
root = tk.Tk()
root.title("Heart Disease Prediction")

# Input fields
tk.Label(root, text="Age").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sex (1=Male, 0=Female)").grid(row=1, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=1, column=1)

tk.Label(root, text="Chest Pain Type (0-3)").grid(row=2, column=0)
entry_cp = tk.Entry(root)
entry_cp.grid(row=2, column=1)

tk.Label(root, text="Resting Blood Pressure").grid(row=3, column=0)
entry_trestbps = tk.Entry(root)
entry_trestbps.grid(row=3, column=1)

tk.Label(root, text="Cholesterol").grid(row=4, column=0)
entry_chol = tk.Entry(root)
entry_chol.grid(row=4, column=1)

tk.Label(root, text="Fasting Blood Sugar (1=True, 0=False)").grid(row=5, column=0)
entry_fbs = tk.Entry(root)
entry_fbs.grid(row=5, column=1)

tk.Label(root, text="Resting ECG (0-2)").grid(row=6, column=0)
entry_restecg = tk.Entry(root)
entry_restecg.grid(row=6, column=1)

tk.Label(root, text="Max Heart Rate Achieved").grid(row=7, column=0)
entry_thalach = tk.Entry(root)
entry_thalach.grid(row=7, column=1)

tk.Label(root, text="Exercise Induced Angina (1=Yes, 0=No)").grid(row=8, column=0)
entry_exang = tk.Entry(root)
entry_exang.grid(row=8, column=1)

tk.Label(root, text="ST Depression Induced by Exercise").grid(row=9, column=0)
entry_oldpeak = tk.Entry(root)
entry_oldpeak.grid(row=9, column=1)

tk.Label(root, text="Slope of the Peak Exercise ST Segment (0-2)").grid(row=10, column=0)
entry_slope = tk.Entry(root)
entry_slope.grid(row=10, column=1)

tk.Label(root, text="Number of Major Vessels (0-3)").grid(row=11, column=0)
entry_ca = tk.Entry(root)
entry_ca.grid(row=11, column=1)

tk.Label(root, text="Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)").grid(row=12, column=0)
entry_thal = tk.Entry(root)
entry_thal.grid(row=12, column=1)

# Predict button
tk.Button(root, text="Predict", command=predict).grid(row=13, column=0, columnspan=2)

root.mainloop()
