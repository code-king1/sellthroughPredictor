from flask import Flask, render_template, request
import tkinter as tk
from tkinter import ttk
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Load Historical Data
data = pd.read_excel("data.xlsx")
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Material', 'Colour', 'Product', 'Category']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Train CatBoost Model
X = data[['Product', 'MRP', 'Material', 'Colour', 'Category']]
y = data['Sell Through']
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
model.fit(X, y, cat_features=[0, 2, 3, 4])  # Pass column indices of categorical features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    product = request.form['product']
    mrp = float(request.form['mrp'])
    material = request.form['material']
    colour = request.form['colour']
    category = request.form['category']

    try:
        # Preprocess user input
        product_encoded = label_encoders['Product'].transform([product])[0]

        # Encode material
        if material in label_encoders['Material'].classes_:
            material_encoded = label_encoders['Material'].transform([material])[0]
        else:
            material_encoded = len(label_encoders['Material'].classes_)

        colour_encoded = label_encoders['Colour'].transform([colour])[0]
        category_encoded = label_encoders['Category'].transform([category])[0]

        # Predict sell-through
        prediction = model.predict([[product_encoded, mrp, material_encoded, colour_encoded, category_encoded]])[0]
        return render_template('result.html', prediction=prediction)
    except ValueError as ve:
        return f"ValueError: {ve}"
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
