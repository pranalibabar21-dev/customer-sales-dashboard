# ============================================
# Customer Sales and Churn Analysis System
# MCA Final Year Major Project
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------
# Chart Style
# --------------------------------------------
sns.set(style="whitegrid")

# --------------------------------------------
# Load Datasets
# --------------------------------------------
sales_data = pd.read_csv("sales_data.csv")
customer_data = pd.read_csv("customer_data.csv")

# --------------------------------------------
# Display First 5 Rows
# --------------------------------------------
print("\n========== SALES DATA ==========")
print(sales_data.head())

print("\n========== CUSTOMER DATA ==========")
print(customer_data.head())

# --------------------------------------------
# Dataset Information
# --------------------------------------------
print("\n========== SALES DATA INFO ==========")
print(sales_data.info())

print("\n========== CUSTOMER DATA INFO ==========")
print(customer_data.info())

# --------------------------------------------
# Check Missing Values
# --------------------------------------------
print("\n========== MISSING VALUES IN SALES DATA ==========")
print(sales_data.isnull().sum())

print("\n========== MISSING VALUES IN CUSTOMER DATA ==========")
print(customer_data.isnull().sum())

# --------------------------------------------
# Remove Duplicates
# --------------------------------------------
sales_data = sales_data.drop_duplicates()
customer_data = customer_data.drop_duplicates()

# --------------------------------------------
# Convert Date Column
# --------------------------------------------
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# --------------------------------------------
# Create Month Column
# --------------------------------------------
sales_data['Month'] = sales_data['Date'].dt.month

# --------------------------------------------
# Dataset Shape
# --------------------------------------------
print("\n========== DATASET SHAPES ==========")
print("Sales Data Shape:", sales_data.shape)
print("Customer Data Shape:", customer_data.shape)

# --------------------------------------------
# Save Cleaned Data
# --------------------------------------------
sales_data.to_csv("cleaned_sales_data.csv", index=False)
customer_data.to_csv("cleaned_customer_data.csv", index=False)

print("\nData Cleaning Completed Successfully!")

# ============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# --------------------------------------------
# Product-wise Sales Analysis
# --------------------------------------------
product_sales = sales_data.groupby('Product')['Total_Sales'].sum()

plt.figure(figsize=(8, 5))
product_sales.plot(kind='bar')

plt.title("Top Selling Products")
plt.xlabel("Product")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

# --------------------------------------------
# Region-wise Sales Distribution
# --------------------------------------------
region_sales = sales_data.groupby('Region')['Total_Sales'].sum()

plt.figure(figsize=(7, 7))
region_sales.plot(kind='pie', autopct='%1.1f%%')

plt.title("Region-wise Sales Distribution")
plt.ylabel("")

plt.tight_layout()
plt.show()

# --------------------------------------------
# Monthly Sales Trend
# --------------------------------------------
monthly_sales = sales_data.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o'
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------
# Top 10 Customers
# --------------------------------------------
top_customers = sales_data.groupby('Customer_ID')['Total_Sales'] \
    .sum() \
    .sort_values(ascending=False) \
    .head(10)

plt.figure(figsize=(10, 5))

top_customers.plot(kind='bar')

plt.title("Top 10 Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

# --------------------------------------------
# Correlation Heatmap
# --------------------------------------------
plt.figure(figsize=(8, 5))

numeric_columns = sales_data.select_dtypes(include=np.number)

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# ============================================
# MACHINE LEARNING - SALES PREDICTION
# ============================================

print("\n========== MACHINE LEARNING MODEL ==========")

# --------------------------------------------
# Features and Target
# --------------------------------------------
X = sales_data[['Quantity', 'Price']]
y = sales_data['Total_Sales']

# --------------------------------------------
# Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# --------------------------------------------
# Create Linear Regression Model
# --------------------------------------------
model = LinearRegression()

# --------------------------------------------
# Train Model
# --------------------------------------------
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# --------------------------------------------
# Make Predictions
# --------------------------------------------
y_pred = model.predict(X_test)

print("\n========== PREDICTED VALUES ==========")
print(y_pred[:10])

# --------------------------------------------
# Model Evaluation
# --------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# --------------------------------------------
# Actual vs Predicted Visualization
# --------------------------------------------
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()

# ============================================
# CHURN ANALYSIS
# ============================================

print("\n========== CHURN ANALYSIS ==========")

churn_count = customer_data['Churn'].value_counts()

plt.figure(figsize=(6, 5))

sns.countplot(
    x='Churn',
    data=customer_data
)

plt.title("Customer Churn Analysis")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("\nChurn Count:")
print(churn_count)

# ============================================
# FINAL INSIGHTS
# ============================================

print("\n========== PROJECT INSIGHTS ==========")

print("""
1. Sales trends were analyzed successfully.
2. Top-performing products were identified.
3. Region-wise revenue distribution was visualized.
4. Monthly sales growth trends were analyzed.
5. Machine Learning model was implemented successfully.
6. Customer churn analysis was completed.
7. Data visualization dashboards were generated successfully.
""")

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")