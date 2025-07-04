import streamlit as st
import pandas as pd
import plotly.express as px
from fetch_data import fetch_flight_data
from process_data import process_flight_data
from summary_ai import summarize_text

# Set up Streamlit page
st.set_page_config(page_title="✈️ Airline Market Demand", layout="wide")
st.title("✈️ Airline Booking Market Demand Insights")

# Button to fetch data
if st.button("Fetch Latest Data"):
    with st.spinner("Fetching flight data..."):
        df = fetch_flight_data()
        st.success("✅ Data fetched and saved to flight_data.csv")

# Try to load and display processed data
try:
    # Process data
    popular_routes = process_flight_data()

    # Bar chart of top routes
    st.subheader("📊 Top Popular Routes")
    fig = px.bar(popular_routes.head(10), x='route', y='count', title='Top 10 Routes by Demand')
    st.plotly_chart(fig)

    # AI Summary
    st.subheader("🧠 AI Summary")
    text = "\n".join([f"{row.route}: {row['count']}" for idx, row in popular_routes.head(10).iterrows()])

    try:
        summary = summarize_text(text)
        st.success(summary)
    except Exception as e:
        st.error("❌ Could not generate summary. Error: " + str(e))

except FileNotFoundError:
    st.warning("⚠️ Please click the 'Fetch Latest Data' button first to get data.")
