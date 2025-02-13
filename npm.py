import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("House Price Prediction")
st.write("This model will take the area and predict the price of the house.")

# Sample data
X = np.array([0, 1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
y = np.array([0, 150000, 200000, 250000, 300000, 350000])

# Train the model
model = LinearRegression()
model.fit(X, y)

# User input
input_size = st.number_input("Enter the size of the house in sqft:", min_value=0)

# Prediction
if input_size == 0:
    st.write(f"Predicted price for {input_size} sqft house: Rs. 0")
else:
    predicted_price = model.predict([[input_size]])
    st.write(f"Predicted price for {input_size} sqft house: Rs. {predicted_price[0]:,.2f}")

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Actual Prices")
    ax.plot(X, model.predict(X), color="red", label="Regression Line")
    ax.scatter([input_size], predicted_price, color="green", label="Predicted Price", zorder=5)
    ax.set_xlabel("Size (sqft)")
    ax.set_ylabel("Price (Rs.)")
    ax.legend()

    # Display the plot
    st.pyplot(fig)