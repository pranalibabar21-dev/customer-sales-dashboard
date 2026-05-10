import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide"
)

# --------------------------------
# Title
# --------------------------------
st.title("Customer Sales and Churn Analysis Dashboard")

st.write("MCA Final Year Major Project")

# --------------------------------
# Load Data
# --------------------------------
sales_data = pd.read_csv("sales_data.csv")
customer_data = pd.read_csv("customer_data.csv")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("Filter Options")

selected_region = st.sidebar.selectbox(
    "Select Region",
    sales_data['Region'].unique()
)

# Filter Data
filtered_data = sales_data[
    sales_data['Region'] == selected_region
]

# --------------------------------
# Show Dataset
# --------------------------------
st.subheader("Sales Dataset")

st.dataframe(filtered_data)

# --------------------------------
# Product Sales Chart
# --------------------------------
st.subheader("Product-wise Sales")

product_sales = filtered_data.groupby(
    'Product'
)['Total_Sales'].sum()

fig, ax = plt.subplots()

product_sales.plot(
    kind='bar',
    ax=ax
)

plt.xlabel("Product")
plt.ylabel("Total Sales")

st.pyplot(fig)

# --------------------------------
# Monthly Sales Trend
# --------------------------------
st.subheader("Monthly Sales Trend")

sales_data['Date'] = pd.to_datetime(
    sales_data['Date']
)

sales_data['Month'] = sales_data[
    'Date'
].dt.month

monthly_sales = sales_data.groupby(
    'Month'
)['Total_Sales'].sum()

fig2, ax2 = plt.subplots()

ax2.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o'
)

plt.xlabel("Month")
plt.ylabel("Total Sales")

st.pyplot(fig2)

# --------------------------------
# Correlation Heatmap
# --------------------------------
st.subheader("Correlation Heatmap")

numeric_columns = sales_data.select_dtypes(
    include='number'
)

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax3
)

st.pyplot(fig3)

# --------------------------------
# Machine Learning Model
# --------------------------------
st.subheader("Sales Prediction")

X = sales_data[['Quantity', 'Price']]
y = sales_data['Total_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred)

st.success(f"Model Accuracy (R² Score): {accuracy:.2f}")

# --------------------------------
# Prediction Graph
# --------------------------------
fig4, ax4 = plt.subplots()

ax4.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

st.pyplot(fig4)

# --------------------------------
# Churn Analysis
# --------------------------------
st.subheader("Customer Churn Analysis")

fig5, ax5 = plt.subplots()

sns.countplot(
    x='Churn',
    data=customer_data,
    ax=ax5
)

st.pyplot(fig5)

# --------------------------------
# Footer
# --------------------------------
st.write("Project Developed Using Python and Streamlit")