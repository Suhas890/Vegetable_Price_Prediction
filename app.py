from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and prepare the model
data = pd.read_csv("Vegetable_market.csv")

# Convert 'Month' to numeric format (e.g., "jan" -> 1)
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
data['Month'] = data['Month'].str.lower().map(month_mapping)

# Rename 'Price per kg' for easier reference
data.rename(columns={'Price per kg': 'Price'}, inplace=True)

# Select features and target variable
X = data[['Vegetable', 'Month', 'Season', 'Temp', 'Deasaster Happen in last 3month']]
y = data['Price']

# One-hot encode categorical features, including 'Vegetable'
X = pd.get_dummies(X, columns=['Vegetable', 'Season', 'Deasaster Happen in last 3month'], drop_first=True)

# Handle missing values
X.fillna(X.mean(), inplace=True)
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define a function to predict vegetable price based on input
def predict_price(vegetable, month, season, temp, disaster):
    # Convert month to numeric format
    month_num = month_mapping[month.lower()]

    # Create input data frame
    input_data = pd.DataFrame({
        'Vegetable': [vegetable],
        'Month': [month_num],
        'Temp': [temp],
        'Season': [season],
        'Deasaster Happen in last 3month': [disaster]
    })

    # One-hot encode input data
    input_data = pd.get_dummies(input_data, columns=['Vegetable', 'Season', 'Deasaster Happen in last 3month'], drop_first=True)

    # Ensure all columns match the training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Predict the price
    price = model.predict(input_data)[0]
    return price

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user inputs from the form
        vegetable_input = request.form["vegetable"]
        month_input = request.form["month"]
        season_input = request.form["season"]
        temp_input = float(request.form["temp"])
        disaster_input = request.form["disaster"]

        # Convert disaster_input to 1 for 'yes' and 0 for 'no'
        disaster_input = 1 if disaster_input.lower() == "yes" else 0

        # Call predict_price function
        predicted_price = predict_price(vegetable_input, month_input, season_input, temp_input, disaster_input)
        
        # Return the result to the user
        return render_template("index.html", predicted_price=predicted_price)

    return render_template("index.html", predicted_price=None)

if __name__ == "__main__":
    app.run(debug=True)
