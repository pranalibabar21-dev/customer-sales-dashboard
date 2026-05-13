import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Sales Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------
st.title("Customer Sales and Churn Analysis System")

st.subheader("Login Page")

# Dummy Users
users = {
    "admin": "admin123",
    "user": "user123",
    "Pranali": "pranali"
}

username = st.text_input("Enter Username")

password = st.text_input(
    "Enter Password",
    type="password"
)
username = username.lower()

login_button = st.button("Login")

# ---------------------------------------------------
# LOGIN VALIDATION
# ---------------------------------------------------
if login_button:

    if username in users and users[username] == password:

        st.success(f"Welcome {username}")

        # ---------------------------------------------------
        # LOAD DATA
        # ---------------------------------------------------
        sales_data = pd.read_csv("sales_data.csv")

        customer_data = pd.read_csv("customer_data.csv")

        # ---------------------------------------------------
        # DATE CONVERSION
        # ---------------------------------------------------
        sales_data['Date'] = pd.to_datetime(
            sales_data['Date']
        )

        # ---------------------------------------------------
        # SIDEBAR MENU
        # ---------------------------------------------------
        st.sidebar.title("Navigation")

        menu = st.sidebar.radio(
            "Go To",
            [
                "Dashboard",
                "Sales Analysis",
                "Prediction",
                "Churn Analysis",
                "Reports"
            ]
        )

        # ---------------------------------------------------
        # FILTERS
        # ---------------------------------------------------
        st.sidebar.header("Filter Options")

        selected_region = st.sidebar.selectbox(
            "Select Region",
            sales_data['Region'].unique()
        )

        selected_product = st.sidebar.selectbox(
            "Select Product",
            sales_data['Product'].unique()
        )

        start_date = st.sidebar.date_input(
            "Start Date",
            sales_data['Date'].min()
        )

        end_date = st.sidebar.date_input(
            "End Date",
            sales_data['Date'].max()
        )

        # ---------------------------------------------------
        # FILTER DATA
        # ---------------------------------------------------
        filtered_data = sales_data[
            (sales_data['Region'] == selected_region) &
            (sales_data['Product'] == selected_product) &
            (sales_data['Date'] >= pd.to_datetime(start_date)) &
            (sales_data['Date'] <= pd.to_datetime(end_date))
        ]

        # ---------------------------------------------------
        # KPI METRICS
        # ---------------------------------------------------
        total_sales = filtered_data[
            'Total_Sales'
        ].sum()

        total_customers = filtered_data[
            'Customer_ID'
        ].nunique()

        top_product = filtered_data.groupby(
            'Product'
        )['Total_Sales'].sum().idxmax()

        # ---------------------------------------------------
        # DASHBOARD PAGE
        # ---------------------------------------------------
        if menu == "Dashboard":

            st.header("Dashboard Overview")

            col1, col2, col3 = st.columns(3)

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

            st.subheader("Filtered Dataset")

            st.dataframe(filtered_data)

        # ---------------------------------------------------
        # SALES ANALYSIS PAGE
        # ---------------------------------------------------
        elif menu == "Sales Analysis":

            st.header("Sales Analysis")

            # Product-wise Sales
            st.subheader("Product-wise Sales")

            product_sales = filtered_data.groupby(
                'Product'
            )['Total_Sales'].sum()

            fig1, ax1 = plt.subplots(figsize=(8,5))

            product_sales.plot(
                kind='bar',
                ax=ax1
            )

            plt.xlabel("Product")
            plt.ylabel("Total Sales")

            st.pyplot(fig1)

            # Monthly Sales Trend
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

            # Region-wise Pie Chart
            st.subheader("Region-wise Sales")

            region_sales = sales_data.groupby(
                'Region'
            )['Total_Sales'].sum()

            fig3, ax3 = plt.subplots(figsize=(7,7))

            ax3.pie(
                region_sales,
                labels=region_sales.index,
                autopct='%1.1f%%'
            )

            st.pyplot(fig3)

            # Heatmap
            st.subheader("Correlation Heatmap")

            numeric_columns = sales_data.select_dtypes(
                include='number'
            )

            fig4, ax4 = plt.subplots(figsize=(8,5))

            sns.heatmap(
                numeric_columns.corr(),
                annot=True,
                cmap='coolwarm',
                ax=ax4
            )

            st.pyplot(fig4)

        # ---------------------------------------------------
        # PREDICTION PAGE
        # ---------------------------------------------------
        elif menu == "Prediction":

            st.header("Sales Prediction Using Machine Learning")

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

            accuracy = r2_score(
                y_test,
                y_pred
            )

            st.success(
                f"Model Accuracy (R² Score): {accuracy:.2f}"
            )

            # Prediction Graph
            st.subheader("Actual vs Predicted Sales")

            fig5, ax5 = plt.subplots(figsize=(8,5))

            ax5.scatter(
                y_test,
                y_pred
            )

            plt.xlabel("Actual Sales")

            plt.ylabel("Predicted Sales")

            st.pyplot(fig5)

        # ---------------------------------------------------
        # CHURN ANALYSIS PAGE
        # ---------------------------------------------------
        elif menu == "Churn Analysis":

            st.header("Customer Churn Analysis")

            # Churn Count
            fig6, ax6 = plt.subplots(figsize=(8,5))

            sns.countplot(
                x='Churn',
                data=customer_data,
                ax=ax6
            )

            st.pyplot(fig6)

            # Churn Percentage
            st.subheader("Churn Percentage")

            churn_count = customer_data[
                'Churn'
            ].value_counts()

            fig7, ax7 = plt.subplots(figsize=(7,7))

            ax7.pie(
                churn_count,
                labels=['No Churn', 'Churn'],
                autopct='%1.1f%%'
            )

            st.pyplot(fig7)

        # ---------------------------------------------------
        # REPORTS PAGE
        # ---------------------------------------------------
        elif menu == "Reports":

            st.header("Download Reports")

            csv = filtered_data.to_csv(
                index=False
            )

            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name='sales_report.csv',
                mime='text/csv'
            )

            st.subheader("Business Insights")

            highest_sales_region = sales_data.groupby(
                'Region'
            )['Total_Sales'].sum().idxmax()

            st.info(
                f"Highest sales were generated from the {highest_sales_region} region."
            )

        # ---------------------------------------------------
        # FOOTER
        # ---------------------------------------------------
        st.markdown("---")

        st.write(
            "Project Developed Using Python, Machine Learning and Streamlit"
        )

        st.write(
            "Developed by Pranali Babar"
        )

    else:

        st.error("Invalid Username or Password")
