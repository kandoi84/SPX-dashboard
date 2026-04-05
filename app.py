import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import requests

# ==========================================================
# CONFIG
# ==========================================================
LIMIT_TICKERS = 40  # Start with 40 to see it work, then change to 500
MAX_WORKERS = 10
US_RISK_FREE_RATE = 4.3

st.set_page_config(layout="wide", page_title="S&P 500 Terminal V8", page_icon="🏦")

# ==========================================================
# HELPERS
# ==========================================================
def safe_fmt(v, prefix="", suffix=""):
    if pd.isna(v) or v is None or v == 0: return "N/A"
    try: return f"{prefix}{float(v):,.2f}{suffix}"
    except: return "N/A"

# ==========================================================
# 1. LOAD TICKERS (Wikipedia)
# ==========================================================
@st.cache_data(ttl=86400)
def load_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(io.StringIO(resp.text))[0]
        df = df.rename(columns={"Symbol":"Ticker","Security":"Name","GICS Sector":"Sector"})
        df["Ticker"] = df["Ticker"].str.strip().str.replace(".", "-", regex=False)
        return df[["Ticker", "Name", "Sector"]].head(LIMIT_TICKERS)
    except:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector"])

# ==========================================================
# 2. FETCH DATA (Yahoo Finance Engine - No Key Needed)
# ==========================================================
def get_yahoo_data(t):
    try:
        stock = yf.Ticker(t)
        info = stock.info
        
        # Pulling the "Important Data" for Screener + Deep Dive
        return {
            "Ticker": t,
            "Price": info.get("currentPrice"),
            "PE": info.get("trailingPE"),
            "PB": info.get("priceToBook"),
            "ROE": info.get("returnOnEquity"),
            "MarketCap": info.get("marketCap"),
            "EPS": info.get("trailingEps"),
            "BVPS": info.get("bookValue"),
            "FCF": info.get("freeCashflow"),
            "DivYield": info.get("dividendYield", 0) * 100,
            "Debt_Equity": info.get("debtToEquity"),
            "RevenueGrowth": info.get("revenueGrowth", 0) * 100
        }
    except:
        return {"Ticker": t}

@st.cache_data(ttl=3600) # Data stays fresh for 1 hour
def fetch_all_data(df_list):
    tickers = df_list["Ticker"].tolist()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_yahoo_data, t): t for t in tickers}
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            pct = (i + 1) / len(tickers)
            progress_bar.progress(pct)
            status_text.text(f"Scanning S&P 500: {i+1}/{len(tickers)} stocks loaded...")
            
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================================
# 3. SCORING & VALUATION MATH
# ==========================================================
def add_logic(df):
    if df.empty: return df
    
    # 1. Scoring (0 to 100)
    # Lower PE and higher ROE relative to the loaded set
    df["PE_rank"] = df["PE"].rank(pct=True, ascending=False).fillna(0)
    df["ROE_rank"] = df["ROE"].rank(pct=True).fillna(0)
    df["Score"] = (df["PE_rank"] * 40 + df["ROE_rank"] * 60).round(1)
    
    # 2. Graham Number: sqrt(22.5 * EPS * BVPS)
    def calc_graham(row):
        eps, bvps = row.get("EPS"), row.get("BVPS")
        if pd.notna(eps) and pd.notna(bvps) and eps > 0 and bvps > 0:
            return (22.5 * eps * bvps) ** 0.5
        return None
    df["Graham"] = df.apply(calc_graham, axis=1)
    
    return df

def calc_dcf(fcf, mcap, price, g, r, tg):
    if not fcf or not mcap or not price or price <= 0: return None
    shares = mcap / price
    g_r, r_r, tg_r = g/100, r/100, tg/100
    # 10 Year DCF
    pv = sum(fcf*(1+g_r)**yr/(1+r_r)**yr for yr in range(1, 11))
    tv = (fcf*(1+g_r)**10*(1+tg_r))/(r_r-tg_r)
    pv += tv/(1+r_r)**10
    return pv/shares

# ==========================================================
# MAIN UI
# ==========================================================
st.title("📊 S&P 500 Intelligence Terminal V8")
st.caption("🚀 Switching to the Yahoo Hybrid Engine (No API Key Required)")

ticker_df = load_sp500()
if ticker_df.empty:
    st.error("Could not load S&P 500 list from Wikipedia.")
    st.stop()

raw_df = fetch_all_data(ticker_df)
df = pd.merge(raw_df, ticker_df, on="Ticker", how="inner")
df = add_logic(df)

# Filters
with st.sidebar:
    st.header("⚙️ Dashboard Config")
    sector_list = ["All"] + sorted(df["Sector"].unique().tolist())
    sel_sector = st.selectbox("Industry", sector_list)
    pe_max = st.slider("Max P/E", 0, 100, 100)

fdf = df.copy()
if sel_sector != "All": fdf = fdf[fdf["Sector"] == sel_sector]
fdf = fdf[(fdf["PE"].isna()) | (fdf["PE"] <= pe_max)]

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📋 Screener","🗺️ Quality Map","📈 Price Chart","🏦 Deep Dive Terminal"])

with tab1:
    st.subheader(" Ranked Opportunities")
    st.dataframe(fdf.sort_values("Score", ascending=False)[["Ticker","Name","Score","PE","ROE","DivYield"]], 
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🗺️ Profitability vs. Valuation")
    fig = px.scatter(fdf.dropna(subset=["PE","ROE"]), x="PE", y="ROE", color="Sector", size="MarketCap", hover_name="Ticker", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 Live Historical Performance")
    t_chart = st.selectbox("Select Symbol", fdf["Ticker"].unique())
    if t_chart:
        hist = yf.download(t_chart, period="1y", progress=False)
        fig_p = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], line=dict(color="#1f77b4"))])
        fig_p.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    st.subheader("🕵️ Fundamental Intrinsic Value")
    t_dive = st.selectbox("Analyze Stock", fdf["Ticker"].unique(), key="dive")
    row = fdf[fdf["Ticker"] == t_dive].iloc[0]

    c1, c2, c3 = st.columns(3)
    g_rate = c1.slider("FCF Growth % (10yr)", 0, 30, 10)
    d_rate = c2.slider("Discount Rate %", 5, 20, 10)
    t_rate = c3.slider("Terminal Growth %", 1, 5, 3)

    # Valuation Math
    d_val = calc_dcf(row.get("FCF"), row.get("MarketCap"), row.get("Price"), g_rate, d_rate, t_rate)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Market Price", safe_fmt(row.get('Price'), prefix="$"))
    m2.metric("Graham Fair Value", safe_fmt(row.get('Graham'), prefix="$"))
    m3.metric("DCF Intrinsic Value", safe_fmt(d_val, prefix="$"))
    
    st.markdown(f"""
    **{row['Name']} Fundamental Summary:**
    * **ROE:** {safe_fmt(row.get('ROE'), suffix="%")}
    * **Debt/Equity:** {safe_fmt(row.get('Debt_Equity'))}
    * **Revenue Growth:** {safe_fmt(row.get('RevenueGrowth'), suffix="%")}
    """)
