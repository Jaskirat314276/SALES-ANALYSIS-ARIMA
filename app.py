"""Streamlit dashboard for the SARIMA champagne-demand forecaster."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.stattools import adfuller

CSV = Path(__file__).parent / "perrin-freres-monthly-champagne-.csv"

st.set_page_config(
    page_title="Champagne Demand Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0f1117;
    --panel: #161a23;
    --panel-2: #1c2230;
    --accent: #f5b042;
    --accent-2: #ff7a59;
    --text: #e8ecf1;
    --muted: #8b95a7;
    --border: #232a3a;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(180deg, #0b0d13 0%, #11141c 100%) !important;
    color: var(--text);
}

#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: auto;
}
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: var(--text) !important;
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    padding: 4px;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

.hero {
    background: linear-gradient(135deg, rgba(245,176,66,0.12) 0%, rgba(255,122,89,0.10) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    margin: 0;
    font-size: 0.98rem;
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card h3 {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin: 0 0 12px 0;
}

.stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.stat .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.stat .value {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.stat .delta {
    font-size: 0.78rem;
    color: var(--muted);
}
.stat .delta.good {color: #4ade80;}
.stat .delta.bad {color: #f87171;}

.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.pill.good {background: rgba(74,222,128,0.12); color: #4ade80;}
.pill.bad {background: rgba(248,113,113,0.12); color: #f87171;}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 28px 0 12px 0;
    color: var(--text);
    letter-spacing: -0.01em;
}

[data-testid="stSidebar"] {
    background: #0c0e14 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text) !important;
    font-weight: 500;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    color: #0b0d13;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 18px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(CSV)
    df.columns = ["Month", "demand"]
    df = df.dropna().reset_index(drop=True)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    df.index.freq = "MS"
    return df


@st.cache_resource
def fit_model(p, d, q, P, D, Q, s):
    df = load_data()
    model = sm.tsa.statespace.SARIMAX(
        df["demand"],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
    )
    return model.fit(disp=False)


def stat_card(label, value, delta=None, delta_kind=None):
    delta_html = ""
    if delta:
        cls = f"delta {delta_kind}" if delta_kind else "delta"
        delta_html = f'<div class="{cls}">{delta}</div>'
    return f"""
    <div class="card">
        <div class="stat">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
    </div>
    """


df = load_data()

st.markdown(
    """
    <div class="hero">
        <h1>Champagne Demand Forecaster</h1>
        <p>Seasonal ARIMA modelling on Perrin Frères monthly demand, 1964–1972.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dashboard, tab_guide = st.tabs(["Dashboard", "Guide"])

with st.sidebar:
    st.markdown("### Model parameters")
    st.caption("Non-seasonal order")
    p = st.slider("p — AR", 0, 3, 1)
    d = st.slider("d — diff", 0, 2, 1)
    q = st.slider("q — MA", 0, 3, 1)
    st.caption("Seasonal order")
    P = st.slider("P", 0, 2, 1)
    D = st.slider("D", 0, 2, 1)
    Q = st.slider("Q", 0, 2, 1)
    s = st.number_input("Period (s)", 1, 24, 12)
    st.markdown("---")
    st.markdown("### Forecast")
    horizon = st.slider("Months ahead", 1, 36, 24)

with tab_dashboard:
    peak = df["demand"].max()
    peak_date = df["demand"].idxmax().strftime("%b %Y")
    mean = df["demand"].mean()
    n_obs = len(df)
    raw_p = adfuller(df["demand"])[1]
    seas_p = adfuller((df["demand"] - df["demand"].shift(12)).dropna())[1]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Observations", f"{n_obs}", "monthly, 1964–1972"),
                unsafe_allow_html=True)
    c2.markdown(stat_card("Average demand", f"{mean:,.0f}", "units / month"),
                unsafe_allow_html=True)
    c3.markdown(stat_card("Peak", f"{peak:,.0f}", f"reached {peak_date}"),
                unsafe_allow_html=True)
    c4.markdown(
        stat_card(
            "Seasonality (ADF)",
            f"p = {seas_p:.1e}",
            "stationary after seasonal diff" if seas_p <= 0.05 else "non-stationary",
            "good" if seas_p <= 0.05 else "bad",
        ),
        unsafe_allow_html=True,
    )

    with st.spinner("Fitting SARIMA model..."):
        fit = fit_model(p, d, q, P, D, Q, s)

    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, horizon + 1)]
    forecast = fit.get_forecast(steps=horizon)
    mean_fc = forecast.predicted_mean
    mean_fc.index = future_dates
    ci = forecast.conf_int()
    ci.index = future_dates

    st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["demand"], mode="lines", name="Historical",
            line=dict(color="#7da3ff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(ci.index) + list(ci.index[::-1]),
            y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
            fill="toself",
            fillcolor="rgba(245,176,66,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% confidence",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_fc.index, y=mean_fc, mode="lines", name="Forecast",
            line=dict(color="#f5b042", width=2.5),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161a23",
        plot_bgcolor="#161a23",
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#232a3a", title=""),
        yaxis=dict(gridcolor="#232a3a", title="Demand (units)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Model fit</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(stat_card("Order", f"({p},{d},{q})", f"seasonal ({P},{D},{Q},{s})"),
                unsafe_allow_html=True)
    m2.markdown(stat_card("AIC", f"{fit.aic:,.1f}", "lower = better"), unsafe_allow_html=True)
    m3.markdown(stat_card("BIC", f"{fit.bic:,.1f}", "lower = better"), unsafe_allow_html=True)
    m4.markdown(stat_card("Log-likelihood", f"{fit.llf:,.1f}"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecast table</div>', unsafe_allow_html=True)
    out = pd.DataFrame({
        "forecast": mean_fc.round(1),
        "lower_95": ci.iloc[:, 0].round(1),
        "upper_95": ci.iloc[:, 1].round(1),
    })
    out.index.name = "month"
    st.dataframe(out, use_container_width=True, height=320)
    st.download_button(
        "Download forecast as CSV",
        out.to_csv().encode(),
        file_name="forecast.csv",
        mime="text/csv",
    )

    with st.expander("Full statsmodels model summary"):
        st.text(str(fit.summary()))


with tab_guide:
    st.markdown('<div class="section-title">What this app does</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This dashboard fits a **Seasonal ARIMA (SARIMA)** model to 9 years of
        monthly champagne sales data (Perrin Frères, 1964–1972) and projects
        demand into the future. Champagne sales explode every December and
        crash every August, so a *seasonal* model is essential — a plain ARIMA
        would smooth that pattern away.

        Use it to play with model orders, see how AIC / BIC react, and download
        the resulting forecast as CSV.
        """
    )

    st.markdown('<div class="section-title">Quick start</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. The default order **SARIMA(1,1,1)(1,1,1,12)** already gives a strong
           fit — start with the dashboard as-is.
        2. Use the sidebar slider **Months ahead** to choose a forecast horizon
           (1–36 months). The chart and table refresh instantly.
        3. Click **Download forecast as CSV** to save the numeric output.
        4. Expand **Full statsmodels model summary** at the bottom for the raw
           coefficient table, p-values, and Ljung-Box diagnostics.
        """
    )

    st.markdown('<div class="section-title">Reading the dashboard</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        - **KPI strip (top row).** Sample size, mean demand, the all-time peak
          month, and the **ADF p-value** on the seasonally-differenced series.
          A p-value ≤ 0.05 means the differenced series is stationary — the
          prerequisite for ARIMA-family models. The default settings achieve
          p ≈ 2 × 10⁻¹¹.
        - **Forecast chart.** Blue is observed history; orange is the model's
          prediction; the shaded orange band is the **95% confidence interval**
          — the model's honest uncertainty. The band widens as you forecast
          further out.
        - **Model fit row.** The chosen order plus three goodness-of-fit
          numbers. **AIC** and **BIC** penalise model complexity; lower is
          better. Compare orders by toggling sliders and watching these move.
        - **Forecast table.** Per-month point forecast plus the lower/upper
          95% bounds. Same data as the chart, in tabular form.
        """
    )

    st.markdown('<div class="section-title">Understanding the parameters</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        SARIMA has two sets of three orders. The non-seasonal `(p, d, q)`
        captures short-term dynamics, and the seasonal `(P, D, Q, s)` captures
        the recurring yearly pattern.

        | Param | Meaning | Typical value |
        |-------|--------------------------------------------------------|---------------|
        | **p** | AR order — how many past values feed the prediction.   | 0–2 |
        | **d** | Differencing — how many times to subtract `series[t-1]` to remove trend. | 1 |
        | **q** | MA order — how many past forecast errors feed the prediction. | 0–2 |
        | **P** | Seasonal AR — past values from the same season last year. | 0–1 |
        | **D** | Seasonal differencing — subtracts `series[t-s]` to remove yearly seasonality. | 1 |
        | **Q** | Seasonal MA — past errors from the same season. | 0–1 |
        | **s** | Length of the seasonal cycle. **12 = monthly with a yearly pattern.** | 12 |

        The defaults `(1,1,1)(1,1,1,12)` are the configuration the original
        notebook converged on. They produce AIC ≈ 1487, a large jump from the
        non-seasonal ARIMA(1,1,1) baseline of AIC ≈ 1912.
        """
    )

    st.markdown('<div class="section-title">Tuning tips</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Always keep `D = 1` and `s = 12` for this dataset.** The yearly
          seasonality is the dominant signal — without seasonal differencing
          the series is non-stationary and the model degrades sharply.
        - **Watch AIC, not the visual fit.** A model that hugs history more
          tightly is often overfit. Lower AIC with simpler orders is usually
          preferable.
        - **Confidence bands grow with the horizon.** At 24+ months the band
          can be wide — that is correct uncertainty quantification, not a bug.
        - **Coefficients with high p-values** in the model summary suggest a
          term is not pulling its weight; consider lowering that order.
        """
    )

    st.markdown('<div class="section-title">Adapting to your own data</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        To run this on a different time series, replace
        `perrin-freres-monthly-champagne-.csv` with a CSV that has two columns:
        a date column and a numeric value column. Adjust `s` to your seasonal
        period (12 for monthly-yearly, 7 for daily-weekly, 24 for hourly-daily,
        4 for quarterly-yearly). Then re-run the app — caching reloads the
        new data automatically.
        """
    )

    st.markdown('<div class="section-title">Running locally</div>', unsafe_allow_html=True)
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash",
    )
    st.caption(
        "For a one-shot, non-interactive run that writes plots and a 24-month "
        "forecast CSV to ./results/, use: python run_sarima.py"
    )
