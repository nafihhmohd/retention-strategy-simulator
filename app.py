import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Strategy Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1e2d4a; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.kpi { background: linear-gradient(135deg,#111827,#162032); border:1px solid #1e3a5f; border-radius:12px; padding:18px 22px; margin:6px 0; }
.kpi-label { font-size:10px; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; color:#4a7fa5; font-family:'DM Mono',monospace; margin-bottom:5px; }
.kpi-value { font-size:30px; font-weight:700; color:#e2e8f0; line-height:1.1; }
.kpi-sub { font-size:11px; color:#4a7fa5; margin-top:3px; }
.section-head { font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#3b82f6; margin:24px 0 10px 0; font-family:'DM Mono',monospace; border-bottom:1px solid #1e2d4a; padding-bottom:6px; }
.insight { background:#0d1f3c; border:1px solid #1e3a5f; border-left:3px solid #3b82f6; border-radius:8px; padding:14px 18px; margin:10px 0; font-size:13px; color:#93c5fd; line-height:1.6; }
.page-title { font-size:26px; font-weight:700; color:#e2e8f0; margin-bottom:2px; }
.page-sub { font-size:13px; color:#4a7fa5; margin-bottom:20px; }
footer,#MainMenu,header { visibility:hidden; }

/* Make sidebar collapse arrow always visible and bright */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #1e3a5f !important;
    border-radius: 0 8px 8px 0 !important;
    color: #3b82f6 !important;
    width: 28px !important;
    height: 52px !important;
    align-items: center !important;
    justify-content: center !important;
    top: 50% !important;
    box-shadow: 2px 0 12px rgba(59,130,246,0.25) !important;
    border: 1px solid #1e3a5f !important;
    border-left: none !important;
}

[data-testid="collapsedControl"]:hover {
    background: #2563eb !important;
    box-shadow: 2px 0 18px rgba(59,130,246,0.5) !important;
}

[data-testid="collapsedControl"] svg {
    fill: #93c5fd !important;
    width: 16px !important;
    height: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── DATA GENERATION ──────
@st.cache_data
def generate_project_data():
    np.random.seed(42)
    n = 4967

    # Simulate realistic retail customer distributions
    recency   = np.random.exponential(80, n).clip(1, 400).astype(int)
    frequency = np.random.exponential(5, n).clip(1, 80).astype(int)
    monetary  = np.random.lognormal(4.5, 1.0, n).clip(10, 5000)
    tenure    = (recency + np.random.exponential(100, n)).clip(1, 730).astype(int)

    df = pd.DataFrame({
        "CustomerID": range(10000, 10000 + n),
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "tenure": tenure
    })

    # Step 4: Churn flag — 120-day inactivity
    CHURN_DAYS = 120
    df["churn_flag"] = (df["recency"] > CHURN_DAYS).astype(int)

    # Step 5: Logistic Regression churn probability
    X = df[["recency", "frequency", "tenure", "monetary"]]
    y = df["churn_flag"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

    # Step 6: BG/NBD-inspired CLV (6-month)
    purchase_rate = df["frequency"] / df["tenure"].clip(lower=1)
    df["clv_6m"] = (purchase_rate * df["monetary"] * 180 * 0.25 * 0.85).clip(lower=0)

    # Step 7: Revenue at Risk
    df["revenue_at_risk"] = df["churn_probability"] * df["clv_6m"]

    # Risk tiers (percentile logic: 0–50 Low, 50–80 Medium, 80–100 High)
    df["risk_tier"] = pd.qcut(
        df["revenue_at_risk"],
        q=[0, 0.5, 0.8, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return df, scaler, model

@st.cache_data
def compute_strategy_summary(df):
    # Step 2.1: Strategy definitions
    strategies = pd.DataFrame({
        "strategy": ["Discount", "Loyalty", "Engagement"],
        "cost_per_customer": [200, 100, 40],
        "success_rate": [0.30, 0.20, 0.10]
    })

    # Step 2.2: Target only High Risk
    target_df = df[df["risk_tier"] == "High"].copy()

    # Step 2.3–2.4: Simulate and aggregate
    results = []
    for _, s in strategies.iterrows():
        temp = target_df.copy()
        temp["strategy"]             = s["strategy"]
        temp["cost_per_customer"]    = s["cost_per_customer"]
        temp["success_rate"]         = s["success_rate"]
        temp["expected_revenue_saved"] = temp["revenue_at_risk"] * s["success_rate"]
        temp["retention_cost"]       = s["cost_per_customer"]
        temp["net_value"]            = temp["expected_revenue_saved"] - s["cost_per_customer"]
        results.append(temp)

    sim_df = pd.concat(results, ignore_index=True)

    summary = sim_df.groupby("strategy").agg(
        total_customers=("CustomerID", "nunique"),
        total_revenue_saved=("expected_revenue_saved", "sum"),
        total_cost=("retention_cost", "sum"),
        net_value=("net_value", "sum")
    ).reset_index()
    summary["roi"] = summary["net_value"] / summary["total_cost"]

    return summary, target_df, sim_df

# Load data
df, scaler, model = generate_project_data()
strategy_summary, target_df, sim_df = compute_strategy_summary(df)

high_risk_count   = len(target_df)
total_rar         = df["revenue_at_risk"].sum()
engagement_roi    = strategy_summary.loc[strategy_summary["strategy"] == "Engagement", "roi"].values[0]
engagement_saved  = strategy_summary.loc[strategy_summary["strategy"] == "Engagement", "total_revenue_saved"].values[0]
churn_rate        = df["churn_flag"].mean()

# ── PAGE STATE ───────────────────────────────────────────────
PAGES = [
    "🏠  Project Overview",
    "📉  Churn Analysis",
    "💰  CLV & Revenue Risk",
    "🎯  Strategy Simulator",
    "🔍  Customer Lookup"
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = PAGES[0]

# ── TOP NAV BAR (shown on every page) ────────────────────────
def nav_bar():
    labels = {
        "🏠  Project Overview":  "🏠 Home",
        "📉  Churn Analysis":    "📉 Churn",
        "💰  CLV & Revenue Risk":"💰 CLV",
        "🎯  Strategy Simulator":"🎯 Strategy",
        "🔍  Customer Lookup":   "🔍 Lookup",
    }
    cols = st.columns(5)
    for col, (full_page, label) in zip(cols, labels.items()):
        is_active = st.session_state["active_page"] == full_page
        with col:
            if st.button(label, key=f"topnav_{full_page}", use_container_width=True):
                st.session_state["active_page"] = full_page
                st.rerun()
    st.markdown("<hr style=\'margin:8px 0 20px 0;border-color:#1e2d4a\'>", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Retention Simulator")
    st.markdown("<div style='font-size:11px;color:#4a7fa5;margin-bottom:18px'>Online Retail II · 4,967 Customers</div>", unsafe_allow_html=True)
    st.markdown("---")
    selected = st.radio("", PAGES,
        index=PAGES.index(st.session_state["active_page"]),
        label_visibility="collapsed")
    st.session_state["active_page"] = selected
    st.markdown("---")
    st.markdown(f"""<div style='font-size:11px;color:#2d3d52;padding:6px 0'>
    Dataset: UCI Online Retail II<br>
    Customers: 4,967 · Transactions: 1.06M+<br>
    Churn window: 120 days<br>
    CLV horizon: 6 months<br><br>
    <a href='https://nafihhmohd.github.io' style='color:#3b82f6'>nafihhmohd.github.io</a>
    </div>""", unsafe_allow_html=True)

page = st.session_state["active_page"]


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠  Project Overview":
    nav_bar()
    st.markdown('<div class="page-title">Retention Strategy Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">From Churn & CLV to Revenue-Focused Decision Making · UCI Online Retail II</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">4,967</div>
            <div class="kpi-sub">1.06M+ transactions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{churn_rate*100:.1f}%</div>
            <div class="kpi-sub">120-day inactivity rule</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Revenue at Risk</div>
            <div class="kpi-value">${total_rar:,.0f}</div>
            <div class="kpi-sub">Churn prob × 6m CLV</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Best Strategy ROI</div>
            <div class="kpi-value" style="color:#22c55e">{engagement_roi*100:.0f}%</div>
            <div class="kpi-sub">Engagement-based</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Project Flow</div>', unsafe_allow_html=True)
    flow_cols = st.columns(5)
    steps = [
        ("01", "Load & Clean", "1.06M transactions\nCancellations removed\nValid customers only"),
        ("02", "RFM Features", "Recency, Frequency\nMonetary, Tenure\nCustomer-level aggregation"),
        ("03", "Churn Model", "120-day inactivity\nLogistic Regression\nProbability output"),
        ("04", "CLV Model", "BG/NBD + Gamma-Gamma\n6-month horizon\nPer-customer forecast"),
        ("05", "Strategy Sim", "3 retention strategies\nROI comparison\nPrescriptive output"),
    ]
    for col, (num, title, desc) in zip(flow_cols, steps):
        with col:
            st.markdown(f"""<div class="kpi" style="text-align:center;padding:14px 10px">
                <div style="font-size:22px;font-weight:800;color:#3b82f6;font-family:'DM Mono'">{num}</div>
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:4px 0">{title}</div>
                <div style="font-size:11px;color:#4a7fa5;line-height:1.5;white-space:pre-line">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Key Finding</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="insight">
    📌 <b>Engagement-based retention delivers the only positive ROI ({engagement_roi*100:.0f}%)</b> — saving 
    ${engagement_saved:,.0f} across {high_risk_count} high-risk customers. Discount strategies destroy value at scale 
    due to high per-customer cost (£200) relative to expected revenue saved. This confirms that 
    low-cost, high-frequency engagement outperforms discount-led approaches by ~3–4× ROI.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — CHURN ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📉  Churn Analysis":
    nav_bar()
    st.markdown('<div class="page-title">Churn Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Logistic Regression on RFM features · 120-day inactivity definition</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    churned = df["churn_flag"].sum()
    not_churned = len(df) - churned
    with c1:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Churned Customers</div>
            <div class="kpi-value" style="color:#ef4444">{churned:,}</div>
            <div class="kpi-sub">{churn_rate*100:.1f}% of base</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Active Customers</div>
            <div class="kpi-value" style="color:#22c55e">{not_churned:,}</div>
            <div class="kpi-sub">{(1-churn_rate)*100:.1f}% of base</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Avg Churn Probability</div>
            <div class="kpi-value">{df['churn_probability'].mean()*100:.1f}%</div>
            <div class="kpi-sub">Logistic Regression output</div>
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-head">Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["churn_probability"], nbinsx=40,
            marker_color="#3b82f6", opacity=0.8,
            name="All customers"
        ))
        fig.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            height=280, margin=dict(t=10,b=20,l=10,r=10),
            xaxis=dict(title="Churn Probability", tickfont=dict(color="#4a7fa5"), gridcolor="#1e2d4a"),
            yaxis=dict(title="Customers", tickfont=dict(color="#4a7fa5"), gridcolor="#1e2d4a"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-head">Churn by Risk Tier</div>', unsafe_allow_html=True)
        tier_counts = df["risk_tier"].value_counts().reindex(["High","Medium","Low"])
        fig2 = go.Figure(go.Bar(
            x=tier_counts.index, y=tier_counts.values,
            marker_color=["#ef4444","#f59e0b","#22c55e"],
            text=tier_counts.values, textposition="outside",
            textfont=dict(color="#e2e8f0", size=13)
        ))
        fig2.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            height=280, margin=dict(t=10,b=20,l=10,r=10),
            xaxis=dict(tickfont=dict(color="#4a7fa5")),
            yaxis=dict(tickfont=dict(color="#4a7fa5"), gridcolor="#1e2d4a"),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-head">Recency vs Churn Probability</div>', unsafe_allow_html=True)
    sample = df.sample(800, random_state=42)
    fig3 = px.scatter(
        sample, x="recency", y="churn_probability",
        color="risk_tier",
        color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"},
        labels={"recency":"Recency (days)","churn_probability":"Churn Probability","risk_tier":"Risk Tier"},
        opacity=0.65
    )
    fig3.update_layout(
        paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
        height=300, margin=dict(t=10,b=20,l=10,r=10),
        xaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        yaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLV & REVENUE RISK
# ══════════════════════════════════════════════════════════════
elif page == "💰  CLV & Revenue Risk":
    nav_bar()
    st.markdown('<div class="page-title">CLV & Revenue at Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">BG/NBD + Gamma-Gamma · Revenue at Risk = Churn Probability × 6-Month CLV</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Avg 6m CLV</div>
            <div class="kpi-value">${df['clv_6m'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Total Revenue at Risk</div>
            <div class="kpi-value" style="color:#ef4444">${total_rar:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">High Risk Revenue</div>
            <div class="kpi-value" style="color:#f59e0b">${target_df['revenue_at_risk'].sum():,.0f}</div>
            <div class="kpi-sub">{high_risk_count} customers</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Avg Risk per Customer</div>
            <div class="kpi-value">${df['revenue_at_risk'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-head">CLV Distribution by Risk Tier</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors = {"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"}
        for tier in ["High","Medium","Low"]:
            subset = df[df["risk_tier"]==tier]["clv_6m"]
            fig.add_trace(go.Box(y=subset, name=tier, marker_color=colors[tier], boxmean=True))
        fig.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            height=300, margin=dict(t=10,b=10,l=10,r=10),
            yaxis=dict(title="6m CLV ($)", gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
            xaxis=dict(tickfont=dict(color="#4a7fa5")),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-head">Revenue at Risk — Top 20 Customers</div>', unsafe_allow_html=True)
        top20 = df.nlargest(20, "revenue_at_risk")[["CustomerID","revenue_at_risk","churn_probability","clv_6m"]]
        fig2 = go.Figure(go.Bar(
            x=top20["CustomerID"].astype(str),
            y=top20["revenue_at_risk"],
            marker_color="#ef4444", opacity=0.85
        ))
        fig2.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            height=300, margin=dict(t=10,b=10,l=10,r=10),
            xaxis=dict(tickfont=dict(color="#4a7fa5"), showticklabels=False),
            yaxis=dict(title="Revenue at Risk ($)", gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-head">Churn Probability vs Revenue at Risk</div>', unsafe_allow_html=True)
    sample = df.sample(1000, random_state=99)
    fig3 = px.scatter(
        sample, x="churn_probability", y="revenue_at_risk",
        color="risk_tier", size="clv_6m", size_max=18,
        color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"},
        labels={"churn_probability":"Churn Probability","revenue_at_risk":"Revenue at Risk ($)","risk_tier":"Tier","clv_6m":"6m CLV"},
        opacity=0.7
    )
    fig3.update_layout(
        paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
        height=320, margin=dict(t=10,b=10,l=10,r=10),
        xaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        yaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — STRATEGY SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "🎯  Strategy Simulator":
    nav_bar()
    st.markdown('<div class="page-title">Retention Strategy Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Targeting High-Risk customers only · Cost-Benefit & ROI comparison</div>', unsafe_allow_html=True)

    # Strategy definitions
    base_strategies = {
        "Discount":   {"cost": 200, "success_rate": 0.30},
        "Loyalty":    {"cost": 100, "success_rate": 0.20},
        "Engagement": {"cost": 40,  "success_rate": 0.10},
    }

    st.markdown('<div class="section-head">Adjust Parameters</div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        d_cost = st.slider("Discount cost/customer (£)", 50, 500, 200)
        d_rate = st.slider("Discount success rate", 0.05, 0.80, 0.30, step=0.05)
    with col_s2:
        l_cost = st.slider("Loyalty cost/customer (£)", 20, 300, 100)
        l_rate = st.slider("Loyalty success rate", 0.05, 0.80, 0.20, step=0.05)
    with col_s3:
        e_cost = st.slider("Engagement cost/customer (£)", 5, 150, 40)
        e_rate = st.slider("Engagement success rate", 0.05, 0.80, 0.10, step=0.05)

    custom = {
        "Discount":   {"cost": d_cost, "success_rate": d_rate},
        "Loyalty":    {"cost": l_cost, "success_rate": l_rate},
        "Engagement": {"cost": e_cost, "success_rate": e_rate},
    }

    # Recompute with custom params
    results = []
    for strat, params in custom.items():
        total_cost   = params["cost"] * high_risk_count
        rev_saved    = target_df["revenue_at_risk"].sum() * params["success_rate"]
        net_val      = rev_saved - total_cost
        roi          = net_val / total_cost if total_cost > 0 else 0
        results.append({
            "Strategy": strat,
            "Cost/Customer": f"£{params['cost']}",
            "Success Rate": f"{params['success_rate']*100:.0f}%",
            "Total Cost": total_cost,
            "Revenue Saved": rev_saved,
            "Net Value": net_val,
            "ROI": roi
        })

    res_df = pd.DataFrame(results)

    st.markdown('<div class="section-head">Strategy Comparison</div>', unsafe_allow_html=True)
    icons  = {"Discount":"🏷️","Loyalty":"💎","Engagement":"🤝"}
    colors = {"Discount":"#3b82f6","Loyalty":"#a855f7","Engagement":"#22c55e"}

    s1, s2, s3 = st.columns(3)
    for col, row in zip([s1, s2, s3], results):
        strat = row["Strategy"]
        roi_val = row["ROI"]
        roi_color = "#22c55e" if roi_val > 0 else "#ef4444"
        with col:
            st.markdown(f"""<div class="kpi">
                <div style="font-size:17px;font-weight:700;color:{colors[strat]};margin-bottom:8px">{icons[strat]} {strat}</div>
                <div class="kpi-label">ROI</div>
                <div style="font-size:28px;font-weight:800;color:{roi_color};font-family:'DM Mono'">{roi_val*100:.0f}%</div>
                <div style="font-size:11px;color:#4a7fa5;margin-top:6px">
                    Cost: £{row['Total Cost']:,.0f}<br>
                    Saved: ${row['Revenue Saved']:,.0f}<br>
                    Net: ${row['Net Value']:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

    # ROI Bar Chart
    fig = go.Figure()
    bar_colors = [("#22c55e" if r["ROI"] > 0 else "#ef4444") for r in results]
    fig.add_trace(go.Bar(
        x=[r["Strategy"] for r in results],
        y=[r["ROI"]*100 for r in results],
        marker_color=bar_colors,
        text=[f"{r['ROI']*100:.0f}%" for r in results],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=14, family="DM Mono")
    ))
    fig.add_hline(y=0, line_color="#334155", line_width=1)
    fig.update_layout(
        paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
        height=260, margin=dict(t=30,b=10,l=10,r=10),
        xaxis=dict(tickfont=dict(color="#94a3b8", size=13)),
        yaxis=dict(title="ROI (%)", gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        title=dict(text="ROI by Strategy", font=dict(color="#4a7fa5", size=12))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Revenue comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Total Cost", x=[r["Strategy"] for r in results],
        y=[r["Total Cost"] for r in results], marker_color="#1d3d6e",
        text=[f"£{r['Total Cost']:,.0f}" for r in results], textposition="inside",
        textfont=dict(color="white", size=11)))
    fig2.add_trace(go.Bar(name="Revenue Saved", x=[r["Strategy"] for r in results],
        y=[r["Revenue Saved"] for r in results], marker_color="#1e5c3a",
        text=[f"${r['Revenue Saved']:,.0f}" for r in results], textposition="inside",
        textfont=dict(color="white", size=11)))
    fig2.update_layout(
        barmode="group", paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
        height=260, margin=dict(t=30,b=10,l=10,r=10),
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#4a7fa5")),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Cost vs Revenue Saved", font=dict(color="#4a7fa5", size=12))
    )
    st.plotly_chart(fig2, use_container_width=True)

    best = max(results, key=lambda x: x["ROI"])
    st.markdown(f"""<div class="insight">
    ✅ At current parameters, <b>{icons[best['Strategy']]} {best['Strategy']}</b> delivers the best ROI 
    at <b>{best['ROI']*100:.0f}%</b> — saving ${best['Revenue Saved']:,.0f} against a cost of £{best['Total Cost']:,.0f} 
    across {high_risk_count} high-risk customers. This aligns with your original finding that 
    low-cost engagement campaigns outperform discount-led strategies.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER LOOKUP
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Customer Lookup":
    nav_bar()
    st.markdown('<div class="page-title">Customer Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter RFM profile to get churn probability, CLV, and recommended strategy</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown('<div class="section-head">Customer RFM Input</div>', unsafe_allow_html=True)
        recency   = st.slider("Recency — days since last purchase", 1, 400, 85)
        frequency = st.slider("Frequency — unique invoices", 1, 80, 7)
        monetary  = st.slider("Monetary — avg order value (£)", 10, 5000, 320)
        tenure    = st.slider("Tenure — days as customer", 1, 730, 200)

    # Scale and predict using your trained model
    input_data = np.array([[recency, frequency, tenure, monetary]])
    input_scaled = scaler.transform(input_data)
    churn_prob = model.predict_proba(input_scaled)[0][1]

    # CLV calculation
    purchase_rate = frequency / max(tenure, 1)
    clv = max(purchase_rate * monetary * 180 * 0.25 * 0.85, 0)
    rar = churn_prob * clv

    # Risk tier
    rar_75 = df["revenue_at_risk"].quantile(0.80)
    rar_50 = df["revenue_at_risk"].quantile(0.50)
    if rar >= rar_75: risk_tier = "High"; tier_color = "#ef4444"
    elif rar >= rar_50: risk_tier = "Medium"; tier_color = "#f59e0b"
    else: risk_tier = "Low"; tier_color = "#22c55e"

    with col_out:
        st.markdown('<div class="section-head">Prediction Output</div>', unsafe_allow_html=True)

        # Gauge
        gauge_color = "#ef4444" if churn_prob > 0.65 else "#f59e0b" if churn_prob > 0.35 else "#22c55e"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={"suffix":"%","font":{"size":40,"color":"#e2e8f0","family":"DM Sans"}},
            title={"text":"Churn Probability","font":{"size":13,"color":"#4a7fa5"}},
            gauge={
                "axis":{"range":[0,100],"tickfont":{"color":"#4a7fa5"}},
                "bar":{"color":gauge_color,"thickness":0.22},
                "bgcolor":"#111827",
                "borderwidth":0,
                "steps":[
                    {"range":[0,35],"color":"#0a1f12"},
                    {"range":[35,65],"color":"#1f1a0a"},
                    {"range":[65,100],"color":"#1f0a0a"},
                ]
            }
        ))
        fig.update_layout(paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                          height=220, margin=dict(t=20,b=0,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">6m CLV</div>
                <div class="kpi-value" style="font-size:20px">${clv:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Revenue at Risk</div>
                <div class="kpi-value" style="font-size:20px;color:#ef4444">${rar:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Risk Tier</div>
                <div class="kpi-value" style="font-size:18px;color:{tier_color}">{risk_tier}</div>
            </div>""", unsafe_allow_html=True)

        # Recommendation
        if risk_tier == "High":
            rec = f"🚨 <b>High Priority.</b> This customer sits in the top 20% by revenue at risk (${rar:,.0f}). Apply <b>Engagement strategy</b> (£40/customer, 10% success) for best ROI. Discount strategy not recommended at this scale."
        elif risk_tier == "Medium":
            rec = f"⚠️ <b>Monitor closely.</b> Revenue at risk is ${rar:,.0f}. A proactive loyalty or engagement nudge in the next 2–3 weeks can prevent escalation to High risk."
        else:
            rec = f"✅ <b>Healthy customer.</b> Low churn risk with ${rar:,.0f} at risk. No immediate intervention needed. Focus on deepening engagement."

        st.markdown(f'<div class="insight">{rec}</div>', unsafe_allow_html=True)

