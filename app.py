from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and data
try:
    model = pickle.load(open('RandomForest.pkl', 'rb'))
    car = pd.read_csv('cleaned_car.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure 'RandomForest.pkl' and 'cleaned_car.csv' are in the same directory as this script.")
    exit()

@app.route('/')
def index():
    # Prepare data for dropdowns
    try:
        Year = sorted(car['Year'].unique(), reverse=True)
        Kms_driven = sorted(car['Kms_Driven'].unique())
        Fuel_type = sorted(car['Fuel_Type'].unique())
        Transmission = sorted(car['Transmission'].unique())
        Owner = sorted(car['Owner'].unique())
        Brand = sorted(car['Brand'].unique())
    except KeyError as e:
        print(f"Error: {e}")
        print("Ensure 'cleaned_car.csv' has the required columns.")
        exit()

    # Render the index.html template
    return render_template(
        'index.html',
        Years=Year,
        Kms_driven=Kms_driven,
        Fuel_type=Fuel_type,
        Transmission=Transmission,
        Owner_type=Owner,
        Brands=Brand
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kilo_driven'))
        fuel_type = request.form.get('fuel_type')
        transmission = request.form.get('transmission')
        owner = request.form.get('owner')
        brand = request.form.get('brand')

        # Prepare data for prediction
        input_data = pd.DataFrame([[year, kms_driven, fuel_type, transmission, owner, brand]],
                                  columns=['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Owner', 'Brand'])

        # Make prediction
        prediction = model.predict(input_data)
        return str(np.round(prediction[0], 2))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error during prediction. Please check your inputs."

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
