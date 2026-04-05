import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# =============================================================================
# CONFIG & API
# =============================================================================
# ⚠️ PASTE YOUR BRAND NEW API KEY HERE ⚠️
FMP_API_KEY          = "YOUR_NEW_API_KEY_HERE" 

US_RISK_FREE_RATE    = 4.3    
DCF_DEFAULT_GROWTH   = 10.0
DCF_DEFAULT_DISCOUNT = 10.0
DCF_TERMINAL_GROWTH  = 3.0
DCF_YEARS            = 10
MAX_WORKERS          = 15   
LIMIT_TICKERS        = 5  # 🐌 TEST MODE: Only doing 5 stocks to save your 250 daily limit!

# =============================================================================
# PAGE CONFIG & FORMATTERS
# =============================================================================
st.set_page_config(page_title="S&P 500 Intelligence Dashboard", layout="wide", page_icon="🇺🇸")

def fmt_price(v): return f"${v:,.2f}" if pd.notna(v) and v is not None else "N/A"
def fmt_num(v, d=1): return f"{v:.{d}f}" if pd.notna(v) and v is not None else "N/A"
def fmt_pct(v, d=1): return f"{v:.{d}f}%" if pd.notna(v) and v is not None else "N/A"

# =============================================================================
# 1. LOAD S&P 500 FROM WIKIPEDIA (Bypasses FMP Paywall & 403 Error)
# =============================================================================
@st.cache_data(ttl=86400) # Caches for 24 hours
def load_sp500_tickers() -> pd.DataFrame:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        html_data = requests.get(url, headers=headers, timeout=10).text
        
        # The StringIO fix to stop Pandas from crashing!
        tables = pd.read_html(io.StringIO(html_data))
        df = tables[0]
        
        df = df.rename(columns={"Symbol": "Ticker", "Security": "Name", "GICS Sector": "Sector"})
        df["Ticker"] = df["Ticker"].str.replace('.', '-', regex=False)
        
        return df[["Ticker", "Name", "Sector"]].head(LIMIT_TICKERS)
        
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 list from Wikipedia: {e}")
        st.stop()

# =============================================================================
# 2. MASTER DATA ENGINE (Fast & Reliable)
# =============================================================================
@st.cache_data(ttl=86400)
def fetch_all_data(sp500_df: pd.DataFrame) -> pd.DataFrame:
    tickers = sp500_df["Ticker"].tolist()
    
    # BATCH FETCH QUOTES
    quote_results = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        symbols = ",".join(batch)
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={FMP_API_KEY}"
        try:
            resp = requests.get(url, timeout=10).json()
            if isinstance(resp, list):
                quote_results.extend(resp)
        except Exception: pass
        
    df_quotes = pd.DataFrame(quote_results)
    if df_quotes.empty: return pd.DataFrame()
    
    df_quotes = df_quotes.rename(columns={
        "symbol": "Ticker", "price": "Price", "yearLow": "52W Low", 
        "yearHigh": "52W High", "marketCap": "_mcap", "eps": "_eps", "pe": "PE"
    })

    # DEEP METRICS FETCH
    metric_results = []
    progress = st.progress(0)
    status = st.empty()
    done = 0
    
    def get_metrics(t):
        try:
            m_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{t}?apikey={FMP_API_KEY}"
            res = requests.get(m_url, timeout=5).json()
            if res:
                m = res[0]
                return {
                    "Ticker": t,
                    "PB": m.get("pbRatioTTM"),
                    "PEG": m.get("pegRatioTTM"),
                    "EV/EBITDA": m.get("enterpriseValueOverEBITDATTM"),
                    "ROE (%)": m.get("roeTTM", 0) * 100 if m.get("roeTTM") else None,
                    "FCF Yield (%)": m.get("freeCashFlowYieldTTM", 0) * 100 if m.get("freeCashFlowYieldTTM") else None,
                    "Div Yield (%)": m.get("dividendYieldPercentageTTM", 0) * 100 if m.get("dividendYieldPercentageTTM") else None,
                    "_bvps": m.get("bookValuePerShareTTM"),
                    "_d2e": m.get("debtToEquityTTM"),
                    "_fcf_yield_raw": m.get("freeCashFlowYieldTTM", 0)
                }
        except: pass
        return {"Ticker": t}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_metrics, t): t for t in tickers}
        for future in as_completed(futures):
            done += 1
            status.text(f"Fetching metrics {done}/{len(tickers)}: {futures[future]}")
            metric_results.append(future.result())
            progress.progress(done / len(tickers))
            
    status.empty()
    progress.empty()
    df_metrics = pd.DataFrame(metric_results)

    # MERGE & CALCULATE
    df_final = pd.merge(df_quotes, df_metrics, on="Ticker", how="left")
    df_final = pd.merge(df_final, sp500_df, on="Ticker", how="left")
    
    df_final["Mkt Cap ($B)"] = df_final["_mcap"] / 1e9
    df_final["Earn Yield (%)"] = (1 / df_final["PE"] * 100).where(df_final["PE"] > 0, None)
    df_final["% from Low"] = ((df_final["Price"] - df_final["52W Low"]) / df_final["52W Low"] * 100).where(df_final["52W Low"] > 0, None)
    df_final["% from High"] = ((df_final["Price"] - df_final["52W High"]) / df_final["52W High"] * 100).where(df_final["52W High"] > 0, None)
    df_final["_fcf"] = df_final["_fcf_yield_raw"] * df_final["_mcap"]
    
    def calc_graham(row):
        eps, bvps = row.get("_eps"), row.get("_bvps")
        if pd.notna(eps) and pd.notna(bvps) and eps > 0 and bvps > 0:
            return (22.5 * eps * bvps)**0.5
        return None
    df_final["Graham"] = df_final.apply(calc_graham, axis=1)
    
    df_final["Beta"] = None 
    df_final["Rev Growth (%)"] = None 

    return df_final

# =============================================================================
# SCORING & SIGNALS
# =============================================================================
def compute_score(row) -> int:
    score = 0
    if pd.notna(row.get("PE"))             and 0 < row["PE"] < 25:             score += 1
    if pd.notna(row.get("PB"))             and 0 < row["PB"] < 4:              score += 1
    if pd.notna(row.get("PEG"))            and 0 < row["PEG"] < 1:             score += 1
    if pd.notna(row.get("EV/EBITDA"))      and 0 < row["EV/EBITDA"] < 15:      score += 1
    if pd.notna(row.get("ROE (%)"))        and row["ROE (%)"] > 15:             score += 1
    if pd.notna(row.get("FCF Yield (%)"))  and row["FCF Yield (%)"] > 3:        score += 1
    if pd.notna(row.get("% from Low"))     and row["% from Low"] < 25:          score += 1
    if pd.notna(row.get("Earn Yield (%)")) and row["Earn Yield (%)"] > US_RISK_FREE_RATE: score += 1
    return score

MAX_SCORE = 8 

def generate_signals(row) -> str:
    s = []
    if pd.notna(row.get("PE"))             and 0 < row["PE"] < 15:             s.append("Low PE")
    if pd.notna(row.get("PEG"))            and 0 < row["PEG"] < 1:             s.append("PEG<1")
    if pd.notna(row.get("Div Yield (%)"))  and row["Div Yield (%)"] > 2:        s.append("Dividend")
    if pd.notna(row.get("% from Low"))     and row["% from Low"] < 15:          s.append("Near 52W Low")
    if pd.notna(row.get("FCF Yield (%)"))  and row["FCF Yield (%)"] > 5:        s.append("FCF Rich")
    if pd.notna(row.get("ROE (%)"))        and row["ROE (%)"] > 20:             s.append("High ROE")
    return " | ".join(s) if s else "—"

def simple_dcf(fcf, growth_rate, terminal_growth, discount_rate, years, shares):
    if not fcf or fcf <= 0 or not shares or shares <= 0: return None
    g, tg, r = growth_rate / 100, terminal_growth / 100, discount_rate / 100
    pv = sum(fcf * (1 + g)**yr / (1 + r)**yr for yr in range(1, years + 1))
    terminal = (fcf * (1 + g)**years * (1 + tg)) / (r - tg)
    pv += terminal / (1 + r)**years
    return pv / shares

# =============================================================================
# MAIN EXECUTION
# =============================================================================
st.title("🇺🇸 S&P 500 Intelligence Dashboard (FMP Edition)")
st.caption(f"Powered by Financial Modeling Prep • Risk-free rate: {US_RISK_FREE_RATE}%")

sp500_df = load_sp500_tickers()
st.info(f"Loaded **{len(sp500_df)} S&P 500 constituents** for testing. Fetching data via FMP API...")

df = fetch_all_data(sp500_df)

if df.empty:
    st.error("Zero data fetched. Please check your API limits or key.")
    st.stop()

df["Score"]   = df.apply(compute_score, axis=1)
df["Signals"] = df.apply(generate_signals, axis=1)

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
with st.sidebar:
    st.header("🔍 Filters")
    sectors = sorted(df["Sector"].dropna().unique())
    sel_sector = st.selectbox("Sector", ["All"] + sectors)

    st.divider()
    pe_max   = st.slider("Max P/E",        0, 200, 200)
    pb_max   = st.slider("Max P/B",        0, 50,  50)
    ev_max   = st.slider("Max EV/EBITDA",  0, 100, 100)
    low_max  = st.slider("Max % from Low", 0, 500, 500)

    st.divider()
    min_score = st.slider(f"Min Score (out of {MAX_SCORE})", 0, MAX_SCORE, 0)

fdf = df.copy()
if sel_sector != "All": fdf = fdf[fdf["Sector"] == sel_sector]

fdf = fdf[
    (fdf["PE"].isna()        | (fdf["PE"]        <= pe_max))  &
    (fdf["PB"].isna()        | (fdf["PB"]        <= pb_max))  &
    (fdf["EV/EBITDA"].isna() | (fdf["EV/EBITDA"] <= ev_max))  &
    (fdf["% from Low"].isna()| (fdf["% from Low"]<= low_max)) &
    (fdf["Score"]            >= min_score)
]

# =============================================================================
# TOP METRICS
# =============================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies",      len(fdf))
c2.metric("Avg P/E",        fmt_num(fdf["PE"].mean()))
c3.metric("Avg EV/EBITDA",  fmt_num(fdf["EV/EBITDA"].mean()))
c4.metric("Avg ROE",        fmt_pct(fdf["ROE (%)"].mean()))
c5.metric("Avg % from Low", fmt_pct(fdf["% from Low"].mean()))
st.divider()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Screener", "🗺️ Valuation Map", "🏭 Sector Analysis", "📈 Price Chart", "🏦 Deep Dive Terminal"])

# --------------------------------------------------------------------------- #
# TAB 1 — SCREENER
# --------------------------------------------------------------------------- #
with tab1:
    sort_col = st.selectbox("Sort by", ["Score", "PE", "PEG", "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "% from Low", "Mkt Cap ($B)"], index=0)
    sort_asc = st.checkbox("Ascending", value=False)
    sorted_df = fdf.sort_values(sort_col, ascending=sort_asc, na_position="last")

    display_cols = ["Ticker", "Name", "Sector", "Price", "Mkt Cap ($B)", "PE", "PB", "PEG",
                    "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "Div Yield (%)",
                    "% from Low", "% from High", "Score", "Signals"]

    disp = sorted_df[display_cols].copy()
    disp["Price"]          = disp["Price"].apply(fmt_price)
    disp["Mkt Cap ($B)"]   = disp["Mkt Cap ($B)"].apply(lambda x: fmt_num(x, 1))
    disp["PE"]             = disp["PE"].apply(fmt_num)
    disp["PB"]             = disp["PB"].apply(fmt_num)
    disp["PEG"]            = disp["PEG"].apply(fmt_num)
    disp["EV/EBITDA"]      = disp["EV/EBITDA"].apply(fmt_num)
    disp["ROE (%)"]        = disp["ROE (%)"].apply(fmt_pct)
    disp["FCF Yield (%)"]  = disp["FCF Yield (%)"].apply(fmt_pct)
    disp["Div Yield (%)"]  = disp["Div Yield (%)"].apply(fmt_pct)
    disp["% from Low"]     = disp["% from Low"].apply(fmt_pct)
    disp["% from High"]    = disp["% from High"].apply(fmt_pct)

    def color_score(val):
        try:
            v = int(val)
            if v >= 6: return "background-color:#198754;color:white"
            if v >= 4: return "background-color:#ffc107;color:black"
            if v <= 2: return "background-color:#dc3545;color:white"
        except: pass
        return ""

    style_fn = disp.style.map if hasattr(disp.style, "map") else disp.style.applymap
    styled = style_fn(color_score, subset=["Score"])
    st.dataframe(styled, width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
# TAB 2 & 3 — MAP & SECTOR
# --------------------------------------------------------------------------- #
with tab2:
    map_df = fdf.dropna(subset=["PE", "% from Low", "Mkt Cap ($B)"]).copy()
    map_df = map_df[(map_df["PE"] > 0) & (map_df["PE"] < 80)]
    if not map_df.empty:
        fig_map = px.scatter(map_df, x="% from Low", y="PE", size="Mkt Cap ($B)", color="Sector", hover_name="Ticker", template="plotly_white")
        st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    sec_agg = fdf.groupby("Sector").agg({"Ticker":"count", "PE":"mean", "EV/EBITDA":"mean", "ROE (%)":"mean", "Score":"mean"}).reset_index()
    st.dataframe(sec_agg, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# TAB 4 — PRICE CHART 
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("📈 FMP Historical Price Chart")
    chart_choice = st.selectbox("Select Stock", fdf.sort_values("Ticker")["Ticker"].tolist(), key="chart_sel")
    
    if chart_choice:
        c_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{chart_choice}?timeseries=252&apikey={FMP_API_KEY}"
        hist_data = requests.get(c_url).json()
        
        if "historical" in hist_data:
            hist = pd.DataFrame(hist_data["historical"])
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date").set_index("date")
            close = hist["close"]
            avg = close.mean()
            
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=hist.index, y=close, line=dict(color="#1f77b4", width=2.5), name="Price"))
            fig_c.add_hline(y=avg, line_dash="dash", line_color="orange", annotation_text=f"1Y Mean: ${avg:,.2f}")
            fig_c.update_layout(height=450, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_c, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 5 — DEEP DIVE TERMINAL
# --------------------------------------------------------------------------- #
with tab5:
    st.subheader("🕵️ Fundamental Terminal")
    term_choice = st.selectbox("Select Stock", fdf.sort_values("Ticker")["Ticker"].tolist(), key="term_sel")

    if term_choice:
        row = fdf[fdf["Ticker"] == term_choice].iloc[0]
        price   = row.get("Price")
        eps     = row.get("_eps", 0)
        bvps    = row.get("_bvps", 0)
        pe      = row.get("PE")
        mcap    = row.get("_mcap", 1)
        d2e     = row.get("_d2e")
        roe     = row.get("ROE (%)", 0) / 100 if pd.notna(row.get("ROE (%)")) else 0
        pb      = row.get("PB")
        peg     = row.get("PEG")
        graham_number = row.get("Graham")
        fcf     = row.get("_fcf", 0)
        
        graham_margin = ((graham_number - price) / price * 100) if (price and graham_number) else 0
        earn_yield    = row.get("Earn Yield (%)", 0)
        fcf_yield     = row.get("FCF Yield (%)", 0)
        earn_spread   = earn_yield - US_RISK_FREE_RATE if earn_yield else 0

        st.divider()
        st.markdown(f"### {row['Name']} &nbsp;|&nbsp; {row['Sector']}")

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Current Price", fmt_price(price))
        v2.metric("Graham Fair Value", fmt_price(graham_number), f"{graham_margin:.1f}% discount" if graham_margin > 0 else f"{graham_margin:.1f}% premium", delta_color="normal" if graham_margin > 0 else "inverse")
        v3.metric("Earnings Yield", fmt_pct(earn_yield), f"{earn_spread:+.2f}% vs 10Y Treasury" if earn_spread else None, delta_color="normal" if earn_spread > 0 else "inverse")
        v4.metric("FCF Yield", fmt_pct(fcf_yield))

        v5, v6, v7, v8 = st.columns(4)
        v5.metric("P/E", fmt_num(pe))
        v6.metric("P/B", fmt_num(pb))
        v7.metric("PEG", fmt_num(peg))
        v8.metric("Debt / Equity", fmt_num(d2e))

        st.markdown("#### 🔢 DCF Intrinsic Value Estimator")
        dcf1, dcf2, dcf3 = st.columns(3)
        g_rate = dcf1.slider("FCF Growth Rate (%)", 0.0, 30.0, DCF_DEFAULT_GROWTH, 0.5)
        d_rate = dcf2.slider("Discount Rate (%)", 4.0, 20.0, DCF_DEFAULT_DISCOUNT, 0.5)
        t_grw  = dcf3.slider("Terminal Growth Rate (%)", 1.0, 5.0, DCF_TERMINAL_GROWTH, 0.5)

        shares = mcap / price if mcap and price else 1
        dcf_value = simple_dcf(fcf, g_rate, t_grw, d_rate, DCF_YEARS, shares)

        d1, d2, d3 = st.columns(3)
        d1.metric("DCF Intrinsic Value", fmt_price(dcf_value))
        if dcf_value and price:
            dcf_margin = (dcf_value - price) / price * 100
            d2.metric("Margin of Safety", fmt_pct(dcf_margin), delta_color="normal" if dcf_margin > 0 else "inverse")
            d3.metric("Price / DCF", fmt_num(price / dcf_value))

# =============================================================================
# DOWNLOADS
# =============================================================================
st.divider()
dl1, dl2 = st.columns(2)
with dl1:
    export_cols = ["Ticker", "Name", "Sector", "Price", "Mkt Cap ($B)", "PE", "PB", "PEG",
                   "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "Div Yield (%)",
                   "% from Low", "Score", "Signals"]
    st.download_button("⬇️ Download Screener CSV", fdf[export_cols].to_csv(index=False).encode('utf-8'), "sp500_screener.csv", mime="text/csv")
with dl2:
    top10 = fdf.nlargest(10, "Score")[["Ticker", "Name", "Sector", "Score", "Signals"]]
    st.download_button("⬇️ Download Top 10 by Score", top10.to_csv(index=False).encode('utf-8'), "sp500_top10.csv", mime="text/csv")
