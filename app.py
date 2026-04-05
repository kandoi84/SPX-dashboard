import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Pulse Check", layout="wide")
st.title("🩺 Connection Pulse Check")

# A small, safe list of stocks
test_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

st.write("Trying to fetch basic data for:", test_tickers)

results = []

# Snail Mode for the pulse check
for ticker in test_tickers:
    try:
        st.write(f"Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Try to get just the fast info (usually more reliable than .info)
        price = stock.fast_info['last_price']
        
        # Try to get the short name (might fail, so use fallback)
        try:
             name = stock.info.get('shortName', ticker)
        except Exception as e:
             name = f"{ticker} (Name fetch failed: {e})"
             
        results.append({"Ticker": ticker, "Name": name, "Price": price})
        st.success(f"Got data for {ticker}!")
        
    except Exception as e:
        st.error(f"Failed to fetch {ticker}. Error: {e}")

st.divider()

if results:
    st.subheader("Results:")
    df = pd.DataFrame(results)
    st.dataframe(df)
else:
    st.error("Could not fetch any data. This confirms a connection or blocking issue.")
