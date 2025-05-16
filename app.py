# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="AI Finance Advisor", layout="centered")
st.title("💸 AI-Powered Personal Finance Advisor")

# Load and preprocess data
df = pd.read_csv("dataset/sample_transactions.csv")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)  # e.g., '2025-05'

# 📊 Raw Transaction Data
st.subheader("📊 Transaction Data")
st.dataframe(df.head())

# Calculate total spending for May, June, July
df["date"] = pd.to_datetime(df["date"])

df_filtered = df[df["date"].dt.month.isin([5, 6, 7])]
total_spending_may_june_july = df_filtered["amount"].sum()
st.subheader("💰 Total Spending (May - July)")
st.metric(label="Total Spending", value=f"₹{total_spending_may_june_july:,.2f}")


# 📅 Month Selector
st.subheader("📅 Filter by Month")
available_months = sorted(df["month"].unique())
selected_month = st.selectbox("Select a month to analyze:", available_months)

df_month = df[df["month"] == selected_month]

# 🧁 Pie Chart: Category Breakdown (Filtered)
st.subheader(f"🧁 Spending Breakdown by Category ({selected_month})")
category_data = df_month.groupby("category")["amount"].sum()
fig_pie, ax_pie = plt.subplots()
category_data.plot(kind='pie', autopct='%1.1f%%', ax=ax_pie, startangle=90)
ax_pie.set_ylabel("")
st.pyplot(fig_pie)

# 📊 Bar Chart: Monthly Spending Trend
st.subheader("📊 Monthly Expense Trend")
monthly_totals = df.groupby("month")["amount"].sum()
fig_bar, ax_bar = plt.subplots()
monthly_totals.plot(kind="bar", color="skyblue", ax=ax_bar)
ax_bar.set_title("Total Spending per Month")
ax_bar.set_xlabel("Month")
ax_bar.set_ylabel("Amount (₹)")
st.pyplot(fig_bar)

# 📈 Line Chart: Spending Over Time
st.subheader("📈 Daily Spending Over Time")
daily_spending = df.groupby("date")["amount"].sum().sort_index()
st.line_chart(daily_spending)

# 💰 Monthly Budget Tracker
st.subheader("💰 Monthly Budget Tracker")
monthly_budget = st.number_input("Set your monthly budget (₹):", min_value=30000, step=500)

selected_month_date = pd.to_datetime(selected_month + "-01")
monthly_spent = df_month["amount"].sum()

st.metric(label=f"Spent in {selected_month}", value=f"₹{monthly_spent:,.2f}")
if monthly_spent > monthly_budget:
    st.error("🚨 You've exceeded your budget!")
else:
    st.info(f"🧾 Remaining: ₹{monthly_budget - monthly_spent:,.2f}")

# 🔮 Forecasting with Prophet
st.subheader("🔮 Expense Forecast for Next 30 Days")
df_prophet = df[["date", "amount"]].rename(columns={"date": "ds", "amount": "y"})
df_prophet = df_prophet.groupby("ds").sum().reset_index()

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Clean forecast
forecast_clean = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
forecast_clean[["yhat", "yhat_lower", "yhat_upper"]] = forecast_clean[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)
forecast_clean.columns = ["Date", "Predicted Expense (₹)", "Lower Bound (₹)", "Upper Bound (₹)"]

# Plot
fig_forecast = model.plot(forecast)
st.pyplot(fig_forecast)
st.write("📅 Forecasted Daily Expenses (Next 30 Days)")
st.dataframe(forecast_clean)

# 💡 Smart Suggestions
st.subheader("💡 Smart Suggestions")
total_spent = df["amount"].sum()
food_spent = df[df["category"] == "Food"]["amount"].sum()

if food_spent / total_spent > 0.3:
    st.warning("You're spending over 30% on food. Consider cutting down on takeout!")
else:
    st.success("Your food spending is under control. Great job!")

# --- New Suggestions ---

#1. High spending merchant alert
merchant_spending = df.groupby("merchant")["amount"].sum()
top_merchant = merchant_spending.idxmax()
top_merchant_spent = merchant_spending.max()

if top_merchant_spent / total_spent > 0.15:
    st.warning(f"🚩 You spent ₹{top_merchant_spent:,.2f} at {top_merchant}, which is over 15% of your total spending. Consider reviewing these expenses.")
else:
    st.success("Your spending is well-distributed across merchants. Good job!")

#2. Large transaction alert
large_txns = df[df["amount"] > 10000]

if not large_txns.empty:
    st.warning(f"⚠️ You have {len(large_txns)} transaction(s) exceeding ₹10,000. Make sure these are necessary expenses.")
else:
    st.success("No unusually large transactions detected.")
