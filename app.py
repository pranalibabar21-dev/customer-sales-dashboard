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
    page_title="Customer Sales Dashboard",
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
sales_data = pd.read_csv(
    "sales_data.csv"
)

customer_data = pd.read_csv(
    "customer_data.csv"
)

# --------------------------------
# Convert Date Column
# --------------------------------
sales_data['Date'] = pd.to_datetime(
    sales_data['Date']
)

# --------------------------------
# Sidebar Filters
# --------------------------------
st.sidebar.header("Filter Options")

# Region Filter
selected_region = st.sidebar.selectbox(
    "Select Region",
    sales_data['Region'].unique()
)

# Product Filter
selected_product = st.sidebar.selectbox(
    "Select Product",
    sales_data['Product'].unique()
)

# Date Filter
start_date = st.sidebar.date_input(
    "Start Date",
    sales_data['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    sales_data['Date'].max()
)

# --------------------------------
# Filter Data
# --------------------------------
filtered_data = sales_data[
    (sales_data['Region'] == selected_region) &
    (sales_data['Product'] == selected_product) &
    (sales_data['Date'] >= pd.to_datetime(start_date)) &
    (sales_data['Date'] <= pd.to_datetime(end_date))
]

# --------------------------------
# KPI Metrics
# --------------------------------
st.subheader("Dashboard Metrics")

col1, col2, col3 = st.columns(3)

total_sales = filtered_data['Total_Sales'].sum()

total_customers = filtered_data[
    'Customer_ID'
].nunique()

top_product = filtered_data.groupby(
    'Product'
)['Total_Sales'].sum().idxmax()

with col1:
    st.metric(
        "Total Sales",
        f"₹ {total_sales:,}"
    )

with col2:
    st.metric(
        "Total Customers",
        total_customers
    )

with col3:
    st.metric(
        "Top Product",
        top_product
    )

# --------------------------------
# Show Dataset
# --------------------------------
st.subheader("Filtered Sales Dataset")

st.dataframe(filtered_data)

# --------------------------------
# Product-wise Sales Chart
# --------------------------------
st.subheader("Product-wise Sales")

product_sales = filtered_data.groupby(
    'Product'
)['Total_Sales'].sum()

fig, ax = plt.subplots(figsize=(8,5))

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

sales_data['Month'] = sales_data[
    'Date'
].dt.month

monthly_sales = sales_data.groupby(
    'Month'
)['Total_Sales'].sum()

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o'
)

plt.xlabel("Month")
plt.ylabel("Total Sales")

st.pyplot(fig2)

# --------------------------------
# Region-wise Sales Pie Chart
# --------------------------------
st.subheader("Region-wise Sales Distribution")

region_sales = sales_data.groupby(
    'Region'
)['Total_Sales'].sum()

fig_pie, ax_pie = plt.subplots(figsize=(7,7))

ax_pie.pie(
    region_sales,
    labels=region_sales.index,
    autopct='%1.1f%%'
)

st.pyplot(fig_pie)

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
st.subheader("Sales Prediction Using Machine Learning")

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

st.success(
    f"Model Accuracy (R² Score): {accuracy:.2f}"
)

# --------------------------------
# Prediction Graph
# --------------------------------
st.subheader("Actual Sales vs Predicted Sales")

fig4, ax4 = plt.subplots(figsize=(8,5))

ax4.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

st.pyplot(fig4)

# --------------------------------
# Churn Analysis
# --------------------------------
st.subheader("Customer Churn Analysis")

fig5, ax5 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='Churn',
    data=customer_data,
    ax=ax5
)

st.pyplot(fig5)

# --------------------------------
# Churn Percentage
# --------------------------------
st.subheader("Churn Percentage")

churn_count = customer_data[
    'Churn'
].value_counts()

fig6, ax6 = plt.subplots(figsize=(7,7))

ax6.pie(
    churn_count,
    labels=['No Churn', 'Churn'],
    autopct='%1.1f%%'
)

st.pyplot(fig6)

# --------------------------------
# Business Insights
# --------------------------------
st.subheader("Business Insights")

highest_sales_region = sales_data.groupby(
    'Region'
)['Total_Sales'].sum().idxmax()

st.info(
    f"Highest sales were generated from the {highest_sales_region} region."
)

# --------------------------------
# Download Report
# --------------------------------
st.subheader("Download Sales Report")

csv = filtered_data.to_csv(
    index=False
)

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name='sales_report.csv',
    mime='text/csv'
)

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")

st.write(
    "Project Developed Using Python, Machine Learning and Streamlit"
)

st.write(
    "Developed by Pranali Babar"
)
