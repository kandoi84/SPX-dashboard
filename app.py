import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =============================================================================
# CONFIG
# =============================================================================
US_RISK_FREE_RATE  = 4.3    
DCF_DEFAULT_GROWTH   = 10.0
DCF_DEFAULT_DISCOUNT = 10.0
DCF_TERMINAL_GROWTH  = 3.0
DCF_YEARS            = 10
MAX_WORKERS          = 5    # Safe speed limit

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="S&P 500 Intelligence Dashboard", layout="wide", page_icon="🇺🇸")

if "selected_period" not in st.session_state:
    st.session_state.selected_period = "1y"

def fmt_price(v): return f"${v:,.2f}" if pd.notna(v) and v is not None else "N/A"
def fmt_num(v, d=1): return f"{v:.{d}f}" if pd.notna(v) and v is not None else "N/A"
def fmt_pct(v, d=1): return f"{v:.{d}f}%" if pd.notna(v) and v is not None else "N/A"

# =============================================================================
# LOAD S&P 500 TICKERS
# =============================================================================
@st.cache_data(ttl=86400)
def load_sp500_tickers() -> pd.DataFrame:
    st.warning("Using top 50 S&P stocks to guarantee stable loading.")
    fallback = [
        ("AAPL","Apple Inc.","Technology"),("MSFT","Microsoft","Technology"),
        ("NVDA","NVIDIA","Technology"),("AMZN","Amazon","Consumer Cyclical"),
        ("GOOGL","Alphabet A","Communication"),("META","Meta Platforms","Communication"),
        ("TSLA","Tesla","Consumer Cyclical"),("BRK-B","Berkshire Hathaway","Financials"),
        ("JPM","JPMorgan Chase","Financials"),("LLY","Eli Lilly","Healthcare"),
        ("V","Visa","Financials"),("UNH","UnitedHealth","Healthcare"),
        ("XOM","Exxon Mobil","Energy"),("MA","Mastercard","Financials"),
        ("JNJ","Johnson & Johnson","Healthcare"),("PG","Procter & Gamble","Consumer Defensive"),
        ("HD","Home Depot","Consumer Cyclical"),("AVGO","Broadcom","Technology"),
        ("MRK","Merck","Healthcare"),("COST","Costco","Consumer Defensive"),
        ("ABBV","AbbVie","Healthcare"),("CVX","Chevron","Energy"),
        ("CRM","Salesforce","Technology"),("BAC","Bank of America","Financials"),
        ("NFLX","Netflix","Communication"),("AMD","Advanced Micro Devices","Technology"),
        ("PEP","PepsiCo","Consumer Defensive"),("TMO","Thermo Fisher","Healthcare"),
        ("ADBE","Adobe","Technology"),("KO","Coca-Cola","Consumer Defensive"),
        ("WMT","Walmart","Consumer Defensive"),("MCD","McDonald's","Consumer Cyclical"),
        ("CSCO","Cisco","Technology"),("ACN","Accenture","Technology"),
        ("ABT","Abbott","Healthcare"),("ORCL","Oracle","Technology"),
        ("LIN","Linde","Basic Materials"),("INTC","Intel","Technology"),
        ("DHR","Danaher","Healthcare"),("TXN","Texas Instruments","Technology"),
        ("NEE","NextEra Energy","Utilities"),("PM","Philip Morris","Consumer Defensive"),
        ("RTX","Raytheon","Industrials"),("UPS","UPS","Industrials"),
        ("HON","Honeywell","Industrials"),("INTU","Intuit","Technology"),
        ("AMGN","Amgen","Healthcare"),("LOW","Lowe's","Consumer Cyclical"),
        ("IBM","IBM","Technology"), ("QCOM", "Qualcomm", "Technology")
    ]
    return pd.DataFrame(fallback, columns=["Ticker", "Name", "Sector"])

# =============================================================================
# THE VIP PASS (Custom Session to prevent hanging)
# =============================================================================
vip_session = requests.Session()
vip_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

def fetch_single(row: dict) -> dict | None:
    ticker = row["Ticker"]
    try:
        # Pass the VIP session to yfinance so Yahoo doesn't block it
        obj = yf.Ticker(ticker, session=vip_session)
        info = obj.info

        price   = info.get("currentPrice") or info.get("regularMarketPrice")
        pe      = info.get("trailingPE")
        ev      = info.get("enterpriseValue")
        ebitda  = info.get("ebitda")
        roe     = info.get("returnOnEquity")
        
        # Graham Value
        eps = info.get("trailingEps", 0)
        bvps = info.get("bookValue", 0)
        graham = (22.5 * eps * bvps)**0.5 if (eps and bvps and eps > 0 and bvps > 0) else None

        return {
            "Ticker":         ticker,
            "Name":           row["Name"],
            "Sector":         row["Sector"],
            "Price":          price,
            "PE":             pe,
            "EV/EBITDA":      (ev / ebitda) if (ev and ebitda and ebitda != 0) else None,
            "ROE (%)":        (roe * 100) if roe else None,
            "Graham":         graham
        }
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        return None

# =============================================================================
# PARALLEL FETCH (Safe Speed)
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_all(tickers_df: pd.DataFrame) -> pd.DataFrame:
    rows = tickers_df.to_dict("records")
    results = []
    progress = st.progress(0)
    status = st.empty()
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_single, r): r["Ticker"] for r in rows}
        for future in as_completed(futures):
            done += 1
            t = futures[future]
            status.text(f"Fetching {done}/{len(rows)}: {t}")
            res = future.result()
            if res:
                results.append(res)
            progress.progress(done / len(rows))

    status.empty()
    progress.empty()
    return pd.DataFrame(results)

# =============================================================================
# MAIN APP UI
# =============================================================================
st.title("🇺🇸 S&P 500 Intelligence Dashboard")

sp500_df = load_sp500_tickers()
st.info(f"Loaded {len(sp500_df)} stocks. Fetching deep financials safely...")

df = fetch_all(sp500_df)

if df.empty:
    st.error("Zero data fetched. Check your black terminal for errors.")
    st.stop()

# Basic Screen
st.subheader("💰 Valuation Screener")
st.dataframe(df, use_container_width=True, hide_index=True)

# Basic Chart
st.divider()
st.subheader("📈 Price Chart")
target = st.selectbox("Pick a stock", df["Ticker"])
if target:
    hist = yf.Ticker(target, session=vip_session).history(period="1y")
    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Price", line=dict(color="#1f77b4")))
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
