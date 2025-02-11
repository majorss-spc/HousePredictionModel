import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title(" House Price Prediction ")
st.write("This model will Take Area and pridict the price of house ")

X=np.array([0,1000,1500,2000,2500,3000]).reshape(-1,1)
y=np.array([0,150000,200000,250000,300000,350000])

model=LinearRegression()

model.fit(X,y)
# input=int(input("Enter the size of the house : "))
input=st.number_input("Enter the size of the house : ")
if input==0:
    st.write(f"Predicted price for {input} sqft house:Rs. 0")
else:
    predicted_price=model.predict([[input]]) 
    # print(f"Predicted price for 2200 sqft house:${predicted_price[0]:,.2f}")
    st.write(f"Predicted price for {input} sqft house:Rs.{predicted_price[0]:,.2f}")

# plt.scatter(X,y,color="Blue",label="Actual Prices")
# plt.plot(X,model.predict(X),color="red",label="Regression line")
# plt.xlabel("Size(sqft)")
# plt.ylabel("Price($)")
# plt.legend()
# plt.show()