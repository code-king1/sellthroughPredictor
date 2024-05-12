import tkinter as tk
from tkinter import ttk
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load Historical Data
data = pd.read_excel("data.xlsx")

# Preprocess Data
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Material', 'Colour', 'Product', 'Category']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split Data into Training and Testing Sets
X = data[['Product', 'MRP', 'Material', 'Colour', 'Category']]
y = data['Sell Through']

# Train CatBoost Model
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
model.fit(X, y, cat_features=[0, 2, 3, 4])  # Pass column indices of categorical features

# Function to predict sell-through
def predict_sell_through():
    product = product_entry.get()
    mrp = float(mrp_entry.get())
    material = material_entry.get()
    colour = colour_entry.get()
    category = category_entry.get()
    
    try:
        # Preprocess user input
        product_encoded = label_encoders['Product'].transform([product])[0]
        
        # Encode material
        if material in label_encoders['Material'].classes_:
            material_encoded = label_encoders['Material'].transform([material])[0]
        else:
            material_encoded = len(label_encoders['Material'].classes_)  # Assign a new category for unseen labels
        
        colour_encoded = label_encoders['Colour'].transform([colour])[0]
        category_encoded = label_encoders['Category'].transform([category])[0]

        # Predict sell-through
        prediction = model.predict([[product_encoded, mrp, material_encoded, colour_encoded, category_encoded]])[0]
        prediction_label.config(text=f"Predicted Sell Through: {prediction}")
    except ValueError as ve:
        prediction_label.config(text=f"ValueError: {ve}")
    except Exception as e:
        prediction_label.config(text=str(e))

# Create GUI
root = tk.Tk()
root.title("Sell Through Predictor")

# Product
tk.Label(root, text="Product:").grid(row=0, column=0)
product_entry = ttk.Entry(root)
product_entry.grid(row=0, column=1)

# MRP
tk.Label(root, text="MRP:").grid(row=1, column=0)
mrp_entry = ttk.Entry(root)
mrp_entry.grid(row=1, column=1)

# Material
tk.Label(root, text="Material:").grid(row=2, column=0)
material_entry = ttk.Entry(root)
material_entry.grid(row=2, column=1)

# Colour
tk.Label(root, text="Colour:").grid(row=3, column=0)
colour_entry = ttk.Entry(root)
colour_entry.grid(row=3, column=1)

# Category
tk.Label(root, text="Category:").grid(row=4, column=0)
category_entry = ttk.Entry(root)
category_entry.grid(row=4, column=1)

# Button to predict
predict_button = ttk.Button(root, text="Predict", command=predict_sell_through)
predict_button.grid(row=5, columnspan=2)

# Prediction label
prediction_label = ttk.Label(root, text="")
prediction_label.grid(row=6, columnspan=2)

root.mainloop()