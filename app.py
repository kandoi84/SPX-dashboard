import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# =============================================================================
# CONFIG
# =============================================================================
US_RISK_FREE_RATE  = 4.3    # 10Y US Treasury yield (%) — update periodically
DCF_DEFAULT_GROWTH   = 10.0
DCF_DEFAULT_DISCOUNT = 10.0
DCF_TERMINAL_GROWTH  = 3.0
DCF_YEARS            = 10
MAX_WORKERS          = 3    # Lowered to 3 to prevent Yahoo Finance Rate Limiting

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="S&P 500 Intelligence Dashboard", layout="wide", page_icon="🇺🇸")

if "selected_period" not in st.session_state:
    st.session_state.selected_period = "1y"

# =============================================================================
# FORMAT HELPERS
# =============================================================================
def fmt_price(v):
    if pd.isna(v) or v is None: return "N/A"
    return f"${v:,.2f}"

def fmt_num(v, d=1):
    if pd.isna(v) or v is None: return "N/A"
    return f"{v:.{d}f}"

def fmt_pct(v, d=1):
    if pd.isna(v) or v is None: return "N/A"
    return f"{v:.{d}f}%"

def fmt_bn(v):
    """Format market cap in $B"""
    if pd.isna(v) or v is None: return "N/A"
    return f"${v:,.1f}B"

# =============================================================================
# LOAD S&P 500 TICKERS FROM WIKIPEDIA (free, always up to date)
# =============================================================================
@st.cache_data(ttl=86400)  # Refresh once a day
def load_sp500_tickers() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Yahoo Finance screener API.
    Falls back to hardcoded top-50 if API fails.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    # ── Primary: Yahoo Finance SPY holdings screener ─────────────────────────
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {"scrIds": "spy_by_holdings", "count": 600, "formatted": "false"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        quotes = resp.json()["finance"]["result"][0]["quotes"]
        rows = [{"Ticker": q.get("symbol",""), "Name": q.get("shortName", q.get("symbol","")), "Sector": q.get("sector","Unknown")} for q in quotes if q.get("symbol")]
        df = pd.DataFrame(rows).drop_duplicates(subset="Ticker").reset_index(drop=True)
        if not df.empty:
            return df
    except Exception:
        pass

    # ── Fallback: Yahoo Finance large-cap screener ────────────────────────────
    try:
        url2 = "https://query2.finance.yahoo.com/v1/finance/screener"
        params2 = {"formatted": "false", "lang": "en-US", "region": "US"}
        payload = {
            "offset": 0, "size": 600,
            "sortField": "intradaymarketcap", "sortType": "DESC",
            "quoteType": "EQUITY",
            "query": {"operator": "AND", "operands": [
                {"operator": "GT", "operands": ["intradaymarketcap", 10000000000]}
            ]},
        }
        resp2 = requests.post(url2, params=params2, json=payload, headers=headers, timeout=15)
        resp2.raise_for_status()
        quotes2 = resp2.json()["finance"]["result"][0]["quotes"]
        rows2 = [{"Ticker": q.get("symbol",""), "Name": q.get("shortName", q.get("symbol","")), "Sector": q.get("sector","Unknown")} for q in quotes2 if q.get("symbol")]
        df2 = pd.DataFrame(rows2).drop_duplicates(subset="Ticker").reset_index(drop=True)
        if not df2.empty:
            st.warning("Loaded large-cap US stocks (fallback — may not be exact S&P 500 list).")
            return df2
    except Exception:
        pass

    # ── Final fallback: hardcoded top 50 ─────────────────────────────────────
    st.warning("Could not fetch live list — using top 50 fallback.")
    fallback = [
        ("AAPL","Apple Inc.","Technology"),("MSFT","Microsoft","Technology"),
        ("NVDA","NVIDIA","Technology"),("AMZN","Amazon","Consumer Cyclical"),
        ("GOOGL","Alphabet A","Communication"),("GOOG","Alphabet C","Communication"),
        ("META","Meta Platforms","Communication"),("TSLA","Tesla","Consumer Cyclical"),
        ("BRK-B","Berkshire Hathaway","Financials"),("JPM","JPMorgan Chase","Financials"),
        ("LLY","Eli Lilly","Healthcare"),("V","Visa","Financials"),
        ("UNH","UnitedHealth","Healthcare"),("XOM","Exxon Mobil","Energy"),
        ("MA","Mastercard","Financials"),("JNJ","Johnson & Johnson","Healthcare"),
        ("PG","Procter & Gamble","Consumer Defensive"),("HD","Home Depot","Consumer Cyclical"),
        ("AVGO","Broadcom","Technology"),("MRK","Merck","Healthcare"),
        ("COST","Costco","Consumer Defensive"),("ABBV","AbbVie","Healthcare"),
        ("CVX","Chevron","Energy"),("CRM","Salesforce","Technology"),
        ("BAC","Bank of America","Financials"),("NFLX","Netflix","Communication"),
        ("AMD","Advanced Micro Devices","Technology"),("PEP","PepsiCo","Consumer Defensive"),
        ("TMO","Thermo Fisher","Healthcare"),("ADBE","Adobe","Technology"),
        ("KO","Coca-Cola","Consumer Defensive"),("WMT","Walmart","Consumer Defensive"),
        ("MCD","McDonald's","Consumer Cyclical"),("CSCO","Cisco","Technology"),
        ("ACN","Accenture","Technology"),("ABT","Abbott","Healthcare"),
        ("ORCL","Oracle","Technology"),("LIN","Linde","Basic Materials"),
        ("INTC","Intel","Technology"),("DHR","Danaher","Healthcare"),
        ("TXN","Texas Instruments","Technology"),("NEE","NextEra Energy","Utilities"),
        ("PM","Philip Morris","Consumer Defensive"),("RTX","Raytheon","Industrials"),
        ("UPS","UPS","Industrials"),("HON","Honeywell","Industrials"),
        ("INTU","Intuit","Technology"),("AMGN","Amgen","Healthcare"),
        ("LOW","Lowe's","Consumer Cyclical"),("IBM","IBM","Technology"),
    ]
    return pd.DataFrame(fallback, columns=["Ticker", "Name", "Sector"])

# =============================================================================
# FETCH SINGLE TICKER
# =============================================================================
def fetch_single(row: dict) -> dict | None:
    ticker = row["Ticker"]
    try:
        obj  = yf.Ticker(ticker)
        info = obj.info

        price   = info.get("currentPrice") or info.get("regularMarketPrice")
        low52   = info.get("fiftyTwoWeekLow")
        high52  = info.get("fiftyTwoWeekHigh")
        pe      = info.get("trailingPE")
        pb      = info.get("priceToBook")
        peg     = info.get("trailingPegRatio")
        ev      = info.get("enterpriseValue")
        ebitda  = info.get("ebitda")
        fcf     = info.get("freeCashflow")
        mcap    = info.get("marketCap")
        roe     = info.get("returnOnEquity")
        beta    = info.get("beta")
        div_yld = info.get("dividendYield")
        rev_grw = info.get("revenueGrowth")
        eps     = info.get("trailingEps")
        shares  = info.get("sharesOutstanding")

        ev_ebitda   = (ev / ebitda) if (ev and ebitda and ebitda != 0) else None
        pct_low     = ((price - low52)  / low52  * 100) if (price and low52)  else None
        pct_high    = ((price - high52) / high52 * 100) if (price and high52) else None
        mcap_bn     = round(mcap / 1e9, 2) if mcap else None
        fcf_yield   = (fcf / mcap * 100) if (fcf and mcap) else None
        earn_yield  = (1 / pe * 100)      if pe             else None
        div_pct     = (div_yld * 100)     if div_yld        else None
        rev_grw_pct = (rev_grw * 100)     if rev_grw        else None
        roe_pct     = (roe * 100)         if roe            else None

        return {
            "Ticker":         ticker,
            "Name":           row.get("Name", info.get("shortName", ticker)),
            "Sector":         row.get("Sector", info.get("Unknown", "Unknown")),
            "Price":          price,
            "52W Low":        low52,
            "52W High":       high52,
            "Mkt Cap ($B)":   mcap_bn,
            "PE":             pe,
            "PB":             pb,
            "PEG":            peg,
            "EV/EBITDA":      ev_ebitda,
            "ROE (%)":        roe_pct,
            "FCF Yield (%)":  fcf_yield,
            "Earn Yield (%)": earn_yield,
            "Div Yield (%)":  div_pct,
            "Rev Growth (%)": rev_grw_pct,
            "Beta":           beta,
            "% from Low":     pct_low,
            "% from High":    pct_high,
            # Raw for terminal
            "_eps":    eps,
            "_bvps":   info.get("bookValue"),
            "_fcf":    fcf,
            "_shares": shares,
            "_mcap":   mcap,
            "_pe":     pe,
            "_roe":    roe,
            "_d2e":    info.get("debtToEquity"),
            "_pm":     info.get("profitMargins"),
            "_om":     info.get("operatingMargins"),
        }
    except Exception as e:
        # Detective Fix: Print exactly why Yahoo failed to the black terminal box
        print(f"Error fetching {ticker}: {e}") 
        return None

# =============================================================================
# PARALLEL FETCH
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_all(tickers_df: pd.DataFrame) -> pd.DataFrame:
    rows    = tickers_df.to_dict("records")
    results = []
    progress = st.progress(0)
    status   = st.empty()
    done     = 0

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
# SCORING MODEL (calibrated for US market multiples)
# =============================================================================
def compute_score(row) -> int:
    score = 0
    if pd.notna(row.get("PE"))             and 0 < row["PE"] < 25:             score += 1
    if pd.notna(row.get("PB"))             and 0 < row["PB"] < 4:              score += 1
    if pd.notna(row.get("PEG"))            and 0 < row["PEG"] < 1:             score += 1
    if pd.notna(row.get("EV/EBITDA"))      and 0 < row["EV/EBITDA"] < 15:      score += 1
    if pd.notna(row.get("ROE (%)"))        and row["ROE (%)"] > 15:             score += 1
    if pd.notna(row.get("FCF Yield (%)"))  and row["FCF Yield (%)"] > 3:        score += 1
    if pd.notna(row.get("Beta"))           and 0.5 < row["Beta"] < 1.3:         score += 1
    if pd.notna(row.get("% from Low"))     and row["% from Low"] < 25:          score += 1
    if pd.notna(row.get("Rev Growth (%)")) and row["Rev Growth (%)"] > 8:       score += 1
    if pd.notna(row.get("Earn Yield (%)")) and row["Earn Yield (%)"] > US_RISK_FREE_RATE:
        score += 1
    return score

MAX_SCORE = 10

def generate_signals(row) -> str:
    s = []
    if pd.notna(row.get("PE"))             and 0 < row["PE"] < 15:             s.append("Low PE")
    if pd.notna(row.get("PEG"))            and 0 < row["PEG"] < 1:             s.append("PEG<1")
    if pd.notna(row.get("Div Yield (%)"))  and row["Div Yield (%)"] > 2:        s.append("Dividend")
    if pd.notna(row.get("% from Low"))     and row["% from Low"] < 15:          s.append("Near 52W Low")
    if pd.notna(row.get("FCF Yield (%)"))  and row["FCF Yield (%)"] > 5:        s.append("FCF Rich")
    if pd.notna(row.get("ROE (%)"))        and row["ROE (%)"] > 20:             s.append("High ROE")
    if pd.notna(row.get("Rev Growth (%)")) and row["Rev Growth (%)"] > 20:      s.append("Rev Growth")
    if pd.notna(row.get("Beta"))           and row["Beta"] < 0.8:               s.append("Low Beta")
    return " | ".join(s) if s else "—"

# =============================================================================
# DCF MODEL
# =============================================================================
def simple_dcf(fcf, growth_rate, terminal_growth, discount_rate, years, shares):
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return None
    g, tg, r = growth_rate / 100, terminal_growth / 100, discount_rate / 100
    pv = sum(fcf * (1 + g)**yr / (1 + r)**yr for yr in range(1, years + 1))
    terminal = (fcf * (1 + g)**years * (1 + tg)) / (r - tg)
    pv += terminal / (1 + r)**years
    return pv / shares

# =============================================================================
# MAIN
# =============================================================================
st.title("🇺🇸 S&P 500 Intelligence Dashboard")
st.caption(f"Free-data edition · Yahoo Finance + yfinance · Risk-free rate: {US_RISK_FREE_RATE}% (10Y Treasury)")

# Load tickers
sp500_df = load_sp500_tickers()
if sp500_df.empty:
    st.error("Could not load S&P 500 tickers.")
    st.stop()

st.info(f"Loaded **{len(sp500_df)} S&P 500 constituents** from Yahoo Finance. Fetching financials...")

with st.spinner("Fetching market data in parallel — this takes ~2–3 min for all 50 stocks..."):
    df = fetch_all(sp500_df)

if df.empty:
    st.error("Zero data fetched. Please check your terminal for the exact error from Yahoo Finance.")
    st.stop()

df = df.dropna(subset=["Price"])
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
    min_score = st.slider("Min Score (out of 10)", 0, MAX_SCORE, 0)

    st.divider()
    st.caption(f"Risk-free rate: **{US_RISK_FREE_RATE}%** (10Y Treasury)")

fdf = df.copy()
if sel_sector != "All":
    fdf = fdf[fdf["Sector"] == sel_sector]

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Screener",
    "🗺️ Valuation Map",
    "🏭 Sector Analysis",
    "📈 Price Chart",
    "🏦 Deep Dive Terminal",
])

# --------------------------------------------------------------------------- #
# TAB 1 — SCREENER
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("💰 S&P 500 Valuation Screener")

    sort_col = st.selectbox(
        "Sort by",
        ["Score", "PE", "PEG", "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "% from Low", "Mkt Cap ($B)"],
        index=0
    )
    sort_asc = st.checkbox("Ascending", value=False)
    sorted_df = fdf.sort_values(sort_col, ascending=sort_asc, na_position="last")

    display_cols = ["Ticker", "Name", "Sector", "Price", "Mkt Cap ($B)", "PE", "PB", "PEG",
                    "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "Div Yield (%)",
                    "Rev Growth (%)", "Beta", "% from Low", "% from High", "Score", "Signals"]

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
    disp["Rev Growth (%)"] = disp["Rev Growth (%)"].apply(fmt_pct)
    disp["Beta"]           = disp["Beta"].apply(fmt_num)
    disp["% from Low"]     = disp["% from Low"].apply(fmt_pct)
    disp["% from High"]    = disp["% from High"].apply(fmt_pct)

    def color_score(val):
        try:
            v = int(val)
            if v >= 8: return "background-color:#198754;color:white"
            if v >= 5: return "background-color:#ffc107;color:black"
            if v <= 2: return "background-color:#dc3545;color:white"
        except: pass
        return ""

    style_fn = disp.style.map if hasattr(disp.style, "map") else disp.style.applymap
    styled = style_fn(color_score, subset=["Score"])
    st.dataframe(styled, width='stretch', hide_index=True)
    st.caption("Score = composite of valuation, quality & momentum signals (max 10). Green ≥8, Yellow ≥5, Red ≤2.")

# --------------------------------------------------------------------------- #
# TAB 2 — VALUATION MAP
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("🗺️ Valuation Map — PE vs % from 52W Low")
    st.caption("Bottom-left = cheap + near lows = most interesting setups. Bubble size = Market Cap.")

    map_df = fdf.dropna(subset=["PE", "% from Low", "Mkt Cap ($B)"]).copy()
    map_df = map_df[(map_df["PE"] > 0) & (map_df["PE"] < 80)]

    if not map_df.empty:
        fig_map = px.scatter(
            map_df,
            x="% from Low", y="PE",
            size="Mkt Cap ($B)",
            color="Sector",
            hover_name="Ticker",
            hover_data={"Name": True, "PEG": True, "ROE (%)": True, "Score": True},
            title="Valuation Map",
            labels={"% from Low": "% Above 52W Low (lower = closer to lows)",
                    "PE": "P/E Ratio (lower = cheaper)"},
            size_max=60,
            template="plotly_white",
        )
        med_pe  = map_df["PE"].median()
        med_low = map_df["% from Low"].median()
        fig_map.add_hline(y=med_pe,  line_dash="dot", line_color="grey",
                          annotation_text=f"Median PE: {med_pe:.1f}")
        fig_map.add_vline(x=med_low, line_dash="dot", line_color="grey",
                          annotation_text=f"Median % from Low: {med_low:.1f}%")
        fig_map.update_layout(height=580)
        st.plotly_chart(fig_map, width='stretch')

    st.subheader("🎯 Quality Map — PEG vs ROE")
    st.caption("Top-left = high quality at reasonable growth price.")

    qmap = fdf.dropna(subset=["PEG", "ROE (%)", "Mkt Cap ($B)"]).copy()
    qmap = qmap[(qmap["PEG"] > 0) & (qmap["PEG"] < 5) & (qmap["ROE (%)"] > 0)]

    if not qmap.empty:
        fig_q = px.scatter(
            qmap,
            x="PEG", y="ROE (%)",
            size="Mkt Cap ($B)",
            color="Sector",
            hover_name="Ticker",
            hover_data={"Name": True, "PE": True, "Score": True},
            title="Quality Map: PEG vs ROE",
            size_max=60,
            template="plotly_white",
        )
        fig_q.add_vline(x=1,  line_dash="dash", line_color="green",
                        annotation_text="PEG = 1")
        fig_q.add_hline(y=15, line_dash="dash", line_color="blue",
                        annotation_text="ROE = 15%")
        fig_q.update_layout(height=520)
        st.plotly_chart(fig_q, width='stretch')

# --------------------------------------------------------------------------- #
# TAB 3 — SECTOR ANALYSIS
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("🏭 Sector Summary")

    sec_agg = fdf.groupby("Sector").agg(
        Stocks        =("Ticker",        "count"),
        Avg_PE        =("PE",            "mean"),
        Avg_PB        =("PB",            "mean"),
        Avg_EV_EBITDA =("EV/EBITDA",     "mean"),
        Avg_ROE       =("ROE (%)",        "mean"),
        Avg_FCF_Yield =("FCF Yield (%)", "mean"),
        Avg_PctLow    =("% from Low",    "mean"),
        Avg_Score     =("Score",          "mean"),
    ).reset_index()

    sec_disp = sec_agg.copy()
    for col in ["Avg_PE", "Avg_PB", "Avg_EV_EBITDA", "Avg_FCF_Yield", "Avg_PctLow", "Avg_Score"]:
        sec_disp[col] = sec_disp[col].apply(lambda x: fmt_num(x, 1))
    sec_disp["Avg_ROE"] = sec_disp["Avg_ROE"].apply(fmt_pct)
    sec_disp.columns = ["Sector", "# Stocks", "Avg PE", "Avg PB", "Avg EV/EBITDA",
                         "Avg ROE", "Avg FCF Yield (%)", "Avg % from Low", "Avg Score"]
    st.dataframe(sec_disp, width='stretch', hide_index=True)

    fig_bar = px.bar(
        sec_agg.sort_values("Avg_Score", ascending=True),
        x="Avg_Score", y="Sector", orientation="h",
        color="Avg_Score",
        color_continuous_scale="RdYlGn",
        title="Average Composite Score by Sector",
        labels={"Avg_Score": "Avg Score (0–10)"},
        template="plotly_white",
    )
    fig_bar.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, width='stretch')

# --------------------------------------------------------------------------- #
# TAB 4 — PRICE CHART
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("📈 Price Chart with Bollinger Bands")

    p_cols  = st.columns(7)
    periods = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y","MAX":"max"}
    for i, (label, code) in enumerate(periods.items()):
        if p_cols[i].button(label, key=f"period_{label}"):
            st.session_state.selected_period = code
            st.rerun()

    chart_choice = st.selectbox(
        "Select Stock",
        fdf.sort_values("Ticker")["Ticker"].tolist(),
        key="chart_sel",
        format_func=lambda t: f"{t} — {fdf[fdf['Ticker']==t]['Name'].values[0] if not fdf[fdf['Ticker']==t].empty else t}"
    )

    if chart_choice:
        hist = yf.Ticker(chart_choice).history(period=st.session_state.selected_period)

        if not hist.empty:
            close = hist["Close"]
            avg   = close.mean()
            std   = close.std()
            bb_mid = close.rolling(20).mean()
            bb_up  = bb_mid + 2 * close.rolling(20).std()
            bb_dn  = bb_mid - 2 * close.rolling(20).std()

            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=hist.index.tolist() + hist.index.tolist()[::-1],
                y=bb_up.tolist() + bb_dn.tolist()[::-1],
                fill="toself", fillcolor="rgba(100,149,237,0.15)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="Bollinger Band"
            ))
            fig_c.add_trace(go.Scatter(x=hist.index, y=bb_up,  line=dict(color="cornflowerblue", width=1, dash="dot"), name="BB Upper"))
            fig_c.add_trace(go.Scatter(x=hist.index, y=bb_dn,  line=dict(color="cornflowerblue", width=1, dash="dot"), name="BB Lower"))
            fig_c.add_trace(go.Scatter(x=hist.index, y=bb_mid, line=dict(color="grey",           width=1, dash="dash"), name="20MA"))
            fig_c.add_trace(go.Scatter(x=hist.index, y=close,  line=dict(color="#1f77b4",        width=2.5),            name="Price"))
            fig_c.add_trace(go.Bar(
                x=hist.index, y=hist["Volume"], name="Volume",
                marker_color="rgba(150,150,150,0.3)", yaxis="y2"
            ))
            fig_c.add_hline(y=avg, line_dash="dash", line_color="orange",
                            annotation_text=f"Mean: ${avg:,.2f}")
            fig_c.update_layout(
                height=500, template="plotly_white", hovermode="x unified",
                yaxis=dict(title="Price ($)"),
                yaxis2=dict(title="Volume", overlaying="y", side="right",
                            showgrid=False, rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_c, width='stretch')

            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Period High",  fmt_price(close.max()))
            cs2.metric("Period Low",   fmt_price(close.min()))
            cs3.metric("Period Mean",  fmt_price(avg))
            cs4.metric("Volatility",   fmt_pct(std / avg * 100))

# --------------------------------------------------------------------------- #
# TAB 5 — DEEP DIVE TERMINAL
# --------------------------------------------------------------------------- #
with tab5:
    st.subheader("🕵️ Fundamental Terminal")
    st.markdown("Valuation models, quality checks, and intrinsic value estimates — all from free data.")

    term_choice = st.selectbox(
        "Select Stock",
        fdf.sort_values("Ticker")["Ticker"].tolist(),
        key="term_sel",
        format_func=lambda t: f"{t} — {fdf[fdf['Ticker']==t]['Name'].values[0] if not fdf[fdf['Ticker']==t].empty else t}"
    )

    if term_choice:
        with st.spinner(f"Loading deep financials for {term_choice}..."):
            t_obj = yf.Ticker(term_choice)
            info  = t_obj.info

        price   = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
        eps     = info.get("trailingEps",   0) or 0
        bvps    = info.get("bookValue",     0) or 0
        pe      = info.get("trailingPE",    0) or 0
        fcf     = info.get("freeCashflow",  0) or 0
        mcap    = info.get("marketCap",     1) or 1
        d2e     = info.get("debtToEquity")
        roe     = info.get("returnOnEquity", 0) or 0
        shares  = info.get("sharesOutstanding", 1) or 1
        pm      = info.get("profitMargins",  0) or 0
        om      = info.get("operatingMargins",0) or 0
        peg     = info.get("trailingPegRatio")
        pb      = info.get("priceToBook")
        rev_grw = info.get("revenueGrowth",  0) or 0

        graham_number = max(0, 22.5 * eps * bvps) ** 0.5 if (eps > 0 and bvps > 0) else 0
        graham_margin = ((graham_number - price) / price * 100) if (price and graham_number) else 0
        earn_yield    = (1 / pe * 100) if pe else 0
        fcf_yield     = (fcf / mcap * 100) if fcf else 0
        earn_spread   = earn_yield - US_RISK_FREE_RATE

        st.divider()
        st.markdown(f"### {info.get('shortName', term_choice)} &nbsp;|&nbsp; {info.get('sector','N/A')} &nbsp;|&nbsp; {info.get('industry','')}")

        # Valuation
        st.markdown("#### ⚖️ Valuation")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Current Price",     fmt_price(price))
        v2.metric("Graham Fair Value", fmt_price(graham_number),
                  f"{graham_margin:.1f}% {'discount' if graham_margin > 0 else 'premium'}",
                  delta_color="normal" if graham_margin > 0 else "inverse")
        v3.metric("Earnings Yield",    fmt_pct(earn_yield),
                  f"{earn_spread:+.2f}% vs 10Y Treasury",
                  delta_color="normal" if earn_spread > 0 else "inverse")
        v4.metric("FCF Yield",         fmt_pct(fcf_yield))

        v5, v6, v7, v8 = st.columns(4)
        v5.metric("P/E",  fmt_num(pe))
        v6.metric("P/B",  fmt_num(pb))
        v7.metric("PEG",  fmt_num(peg), "Growth-adjusted PE" if peg else None)
        row_data = fdf[fdf["Ticker"] == term_choice]
        v8.metric("Composite Score",
                  f"{compute_score(row_data.iloc[0]) if not row_data.empty else 'N/A'} / {MAX_SCORE}")

        # Quality
        st.markdown("#### 🏥 Quality & Financial Health")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("ROE",              fmt_pct(roe * 100))
        q2.metric("Profit Margin",    fmt_pct(pm * 100))
        q3.metric("Operating Margin", fmt_pct(om * 100))
        q4.metric("Debt / Equity",    fmt_num(d2e) if d2e else "N/A",
                  "× (lower is better)" if d2e else None)

        q5, q6, q7, _ = st.columns(4)
        q5.metric("Revenue Growth", fmt_pct(rev_grw * 100))
        q6.metric("FCF",            f"${fcf/1e9:,.2f}B"  if fcf  else "N/A")
        q7.metric("Market Cap",     f"${mcap/1e9:,.1f}B" if mcap else "N/A")

        # DCF
        st.markdown("#### 🔢 DCF Intrinsic Value Estimator")
        st.caption("Adjust assumptions to stress-test intrinsic value.")
        dcf1, dcf2, dcf3 = st.columns(3)
        g_rate = dcf1.slider("FCF Growth Rate (%)",     0.0, 30.0, DCF_DEFAULT_GROWTH,   0.5, key="dcf_g")
        d_rate = dcf2.slider("Discount Rate (%)",        4.0, 20.0, DCF_DEFAULT_DISCOUNT, 0.5, key="dcf_d")
        t_grw  = dcf3.slider("Terminal Growth Rate (%)", 1.0,  5.0, DCF_TERMINAL_GROWTH,  0.5, key="dcf_t")

        dcf_value = simple_dcf(fcf, g_rate, t_grw, d_rate, DCF_YEARS, shares)

        d1, d2, d3 = st.columns(3)
        d1.metric("DCF Intrinsic Value / Share", fmt_price(dcf_value))
        if dcf_value and price:
            dcf_margin = (dcf_value - price) / price * 100
            d2.metric("Margin of Safety", fmt_pct(dcf_margin),
                      "Undervalued" if dcf_margin > 0 else "Overvalued",
                      delta_color="normal" if dcf_margin > 0 else "inverse")
            d3.metric("Price / DCF",      fmt_num(price / dcf_value))
        else:
            d2.metric("Margin of Safety", "N/A — no FCF data")
            d3.metric("Price / DCF",      "N/A")

        # Analyst notes
        st.markdown("#### 📝 Quick Read")
        notes = []
        if graham_number and price:
            notes.append(f"**Graham Number** (${graham_number:,.2f}) is {'above' if graham_number > price else 'below'} current price — stock appears {'undervalued' if graham_number > price else 'overvalued'} on book + earnings basis.")
        if earn_spread > 0:
            notes.append(f"**Earnings yield** ({earn_yield:.1f}%) exceeds the 10Y Treasury by {earn_spread:.1f}pp — equity risk is compensated.")
        else:
            notes.append(f"**Earnings yield** ({earn_yield:.1f}%) is below the 10Y Treasury — limited risk premium over bonds.")
        if peg and peg < 1:
            notes.append(f"**PEG < 1** ({peg:.2f}) — growth is cheap relative to earnings multiple.")
        elif peg and peg > 2:
            notes.append(f"**PEG > 2** ({peg:.2f}) — growth is expensive; execution must be flawless.")
        if d2e and d2e > 150:
            notes.append(f"**High leverage** (D/E: {d2e:.1f}) — sensitivity to rate movements is elevated.")
        if roe > 0.20:
            notes.append(f"**ROE of {roe*100:.1f}%** signals strong capital efficiency.")
        for note in notes:
            st.markdown(f"- {note}")

        st.info(
            f"**Assumptions:** Risk-free rate = {US_RISK_FREE_RATE}% (10Y US Treasury). "
            f"Graham Number = √(22.5 × EPS × BVPS). "
            f"DCF uses {DCF_YEARS}-year projection. "
            "All data from yfinance — verify with primary sources before acting."
        )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
dl1, dl2 = st.columns(2)
with dl1:
    export_cols = ["Ticker", "Name", "Sector", "Price", "Mkt Cap ($B)", "PE", "PB", "PEG",
                   "EV/EBITDA", "ROE (%)", "FCF Yield (%)", "Div Yield (%)",
                   "Rev Growth (%)", "Beta", "% from Low", "Score", "Signals"]
    st.download_button(
        "⬇️ Download Screener CSV",
        fdf[export_cols].to_csv(index=False),
        "sp500_screener.csv",
        mime="text/csv"
    )
with dl2:
    top10 = fdf.nlargest(10, "Score")[["Ticker", "Name", "Sector", "Score", "Signals"]]
    st.download_button(
        "⬇️ Download Top 10 by Score",
        top10.to_csv(index=False),
        "sp500_top10.csv",
        mime="text/csv"
    )

# =============================================================================
# APPENDIX — METHODOLOGY & CALCULATIONS
# =============================================================================
st.divider()
with st.expander("📚 Appendix: Methodology & Calculations", expanded=False):

    st.markdown("## 📐 Graham Number — Full Explanation")

    st.markdown("""
### What is the Graham Number?

The **Graham Number** is a figure that measures a stock's fundamental value, developed by
**Benjamin Graham** — the father of value investing and mentor to Warren Buffett. It was
introduced in his 1949 book *The Intelligent Investor* and represents the **maximum price
a defensive investor should pay** for a stock based on its earnings and book value.

The idea is simple: a stock trading *below* its Graham Number is potentially undervalued.
A stock trading *above* it may be overpriced relative to its fundamentals.

---

### The Formula
""")

    st.latex(r"\text{Graham Number} = \sqrt{22.5 \times \text{EPS} \times \text{BVPS}}")

    st.markdown("""
Where:
- **EPS** = Earnings Per Share (trailing twelve months)
- **BVPS** = Book Value Per Share (shareholders' equity ÷ shares outstanding)
- **22.5** = a constant derived from Graham's two valuation rules (explained below)

---

### Where Does 22.5 Come From?

Graham set two maximum thresholds for a stock to be considered "reasonably priced":

| Rule | Graham's Threshold | Meaning |
|---|---|---|
| Max P/E ratio | **≤ 15×** | Don't pay more than 15× earnings |
| Max P/B ratio | **≤ 1.5×** | Don't pay more than 1.5× book value |

Multiplying these two limits together: **15 × 1.5 = 22.5**

This means a stock at exactly the Graham Number has a P/E of 15 AND a P/B of 1.5 — 
sitting right at Graham's acceptable upper boundary.

---

### Step-by-Step Worked Example

Let's calculate the Graham Number for a hypothetical stock:

| Input | Value |
|---|---|
| Earnings Per Share (EPS) | $5.00 |
| Book Value Per Share (BVPS) | $40.00 |
""")

    st.latex(r"\text{Graham Number} = \sqrt{22.5 \times \$5.00 \times \$40.00}")
    st.latex(r"= \sqrt{22.5 \times 200}")
    st.latex(r"= \sqrt{4{,}500}")
    st.latex(r"= \$67.08")

    st.markdown("""
So if this stock is trading at **$50**, it is trading at a **25.4% discount** to its
Graham Number — potentially undervalued by Graham's standards.

If it's trading at **$90**, it is trading at a **34.2% premium** — potentially overvalued.

---

### Margin of Safety

Graham always emphasized buying with a **margin of safety** — purchasing significantly
below intrinsic value to protect against estimation errors and market volatility.

The margin of safety shown in this dashboard is calculated as:
""")

    st.latex(r"\text{Margin of Safety} = \frac{\text{Graham Number} - \text{Current Price}}{\text{Current Price}} \times 100")

    st.markdown("""
| Result | Interpretation |
|---|---|
| **Positive %** | Stock is trading below Graham Number → potential discount |
| **Negative %** | Stock is trading above Graham Number → potential premium |
| **> 30% positive** | Strong margin of safety — Graham's preferred range |

---

### Limitations of the Graham Number

The Graham Number is a **starting point**, not a complete valuation. Be aware of:

| Limitation | Why it matters |
|---|---|
| **Requires positive EPS and BVPS** | The formula breaks down for loss-making companies or those with negative book value (common in tech/financials) |
| **Uses trailing EPS** | Backward-looking — doesn't account for high-growth companies whose future earnings will be much higher |
| **Ignores growth** | A company growing at 30% annually may deserve a PE of 30, not 15 — the Graham Number would undervalue it |
| **Book value is distorted** | Share buybacks, goodwill, and intangible assets can make BVPS misleading |
| **Sector blind** | A PE of 15 is expensive for utilities but cheap for software — Graham Number doesn't adjust for sector norms |
| **Inflation era caveat** | Graham developed this in the 1940s–70s with different interest rate regimes |

**Rule of thumb:** Use the Graham Number as a **conservative floor** for valuation, not 
a ceiling. Combine it with DCF, PEG, FCF Yield, and sector comparisons for a fuller picture.

---

### How to Use Graham Number in This Dashboard

1. **Terminal Tab** → Shows Graham Number vs Current Price with discount/premium %
2. **Screener Tab** → Sort by Score; stocks with positive Graham margin score higher
3. **Signals Column** → "Low PE" and "PEG<1" flags complement Graham analysis
4. **Combine with** → DCF (forward-looking), ROE (quality), FCF Yield (cash generation)

---
""")

    st.markdown("## 📊 Other Calculations Used in This Dashboard")

    st.markdown("### Earnings Yield vs Risk-Free Rate")
    st.latex(r"\text{Earnings Yield} = \frac{1}{\text{P/E Ratio}} \times 100")
    st.markdown(f"""
Compared against the **10Y US Treasury yield ({US_RISK_FREE_RATE}%)**.

- If Earnings Yield **>** Risk-Free Rate → equity compensates for risk taken
- If Earnings Yield **<** Risk-Free Rate → bonds may offer better risk-adjusted returns

This is the **Fed Model** of equity valuation, popularized in the 1990s.
""")

    st.markdown("### FCF Yield")
    st.latex(r"\text{FCF Yield} = \frac{\text{Free Cash Flow}}{\text{Market Cap}} \times 100")
    st.markdown("""
Free Cash Flow = Operating Cash Flow − Capital Expenditure

FCF Yield measures how much actual cash a business generates relative to its market value.
A high FCF Yield (>5%) signals the company generates real cash, not just accounting profits.
""")

    st.markdown("### PEG Ratio")
    st.latex(r"\text{PEG} = \frac{\text{P/E Ratio}}{\text{Earnings Growth Rate (\%)}}")
    st.markdown("""
- **PEG < 1** → Stock may be undervalued relative to its growth rate
- **PEG = 1** → Fairly valued (growth matches multiple)
- **PEG > 2** → Growth is expensive; high execution risk

Developed by Peter Lynch. Adjusts P/E for growth — a PE of 30 with 30% growth (PEG=1) 
is arguably cheaper than a PE of 20 with 5% growth (PEG=4).
""")

    st.markdown("### DCF (Discounted Cash Flow)")
    st.latex(r"\text{Intrinsic Value} = \sum_{t=1}^{n} \frac{FCF \times (1+g)^t}{(1+r)^t} + \frac{TV}{(1+r)^n}")
    st.latex(r"\text{Terminal Value (TV)} = \frac{FCF \times (1+g)^n \times (1+g_{terminal})}{r - g_{terminal}}")
    st.markdown(f"""
Where:
- **FCF** = Current Free Cash Flow
- **g** = Projected FCF growth rate (adjustable slider, default {DCF_DEFAULT_GROWTH}%)
- **r** = Discount rate / required return (adjustable slider, default {DCF_DEFAULT_DISCOUNT}%)
- **g_terminal** = Perpetual terminal growth rate (adjustable slider, default {DCF_TERMINAL_GROWTH}%)
- **n** = Projection years ({DCF_YEARS} years)

The DCF result is divided by shares outstanding to get **intrinsic value per share**.

**Key sensitivity:** The discount rate has the largest impact on DCF output. A 1% change 
in discount rate can move intrinsic value by 15–25%. Always run multiple scenarios.
""")

    st.markdown("### Composite Score (0–10)")
    st.markdown(f"""
Each stock is scored on 10 criteria, earning 1 point per criterion passed:

| # | Criterion | Threshold | Rationale |
|---|---|---|---|
| 1 | P/E Ratio | < 25× | Below S&P 500 long-run average |
| 2 | Price/Book | < 4× | Reasonable asset backing |
| 3 | PEG Ratio | < 1× | Growth available at fair price |
| 4 | EV/EBITDA | < 15× | Enterprise value vs operating profit |
| 5 | ROE | > 15% | Quality of capital allocation |
| 6 | FCF Yield | > 3% | Real cash generation |
| 7 | Beta | 0.5–1.3 | Moderate market sensitivity |
| 8 | % from 52W Low | < 25% | Not extended from lows |
| 9 | Revenue Growth | > 8% | Business momentum |
| 10 | Earnings Yield | > {US_RISK_FREE_RATE}% | Beats risk-free rate |

Score ≥ 8 = Strong candidate · Score ≥ 5 = Worth reviewing · Score ≤ 2 = Avoid
""")

    st.info(
        "**Disclaimer:** All calculations use data from yfinance (Yahoo Finance). "
        "This dashboard is for educational and research purposes only — not financial advice. "
        "Always verify figures from primary sources (SEC filings, company reports) "
        "before making any investment decisions."
    )
