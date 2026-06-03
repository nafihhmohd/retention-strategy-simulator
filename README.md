# Retention Strategy Simulator — Churn & CLV Analytics

End-to-end prescriptive analytics system built on the UCI Online Retail II dataset (1.06M+ transactions, 4,967 customers). Predicts customer churn, estimates 6-month CLV using BG/NBD + Gamma-Gamma models, quantifies revenue at risk, and simulates three retention strategies (Discount, Loyalty, Engagement) with ROI comparison.

## Live App
🔗 **[Open Streamlit App](https://retention-strategy-simulator.streamlit.app/)**

## Project Flow

Raw Transactions → RFM Features → Churn Model → CLV Model → Revenue at Risk → Strategy Simulation

| Step | Method | Output |
|------|--------|--------|
| Data Cleaning | Remove cancellations, nulls, invalid rows | 1.06M valid transactions |
| RFM Engineering | Customer-level aggregation | Recency, Frequency, Monetary, Tenure |
| Churn Definition | 120-day inactivity rule | Binary churn flag per customer |
| Churn Probability | Logistic Regression on RFM features | Probability score per customer |
| CLV Estimation | BG/NBD + Gamma-Gamma (lifetimes) | 6-month CLV per customer |
| Revenue at Risk | Churn Probability × 6m CLV | Prioritization metric |
| Risk Tiers | Percentile-based (0–50 Low, 50–80 Medium, 80–100 High) | Targeting logic |
| Strategy Simulation | 3 strategies × High-risk customers | ROI comparison |

## Key Findings

- **Churn rate:** ~55% of customers inactive beyond 120 days
- **Engagement strategy** delivers the only positive ROI (~54%) — cost £40/customer vs £200 for Discount
- **Discount strategy destroys value** at scale — high cost relative to expected revenue saved
- **Revenue recovery potential:** $196K+ identified across high-risk segment

## Retention Strategy Results

| Strategy | Cost/Customer | Success Rate | ROI |
|----------|--------------|--------------|-----|
| Discount | £200 | 30% | Negative |
| Loyalty | £100 | 20% | Negative |
| **Engagement** | **£40** | **10%** | **Positive (~54%)** |

> Engagement-based retention outperforms discount-led strategies by ~3.4× ROI — confirming that low-cost, high-frequency touchpoints are more effective than one-time monetary incentives.

## Tech Stack

| Tool | Usage |
|------|-------|
| Python | Data processing, modeling |
| Pandas / NumPy | RFM engineering, data cleaning |
| Scikit-learn | Logistic Regression churn model |
| Lifetimes | BG/NBD + Gamma-Gamma CLV modeling |
| Streamlit | Interactive web app |
| Plotly | Visualizations and dashboards |
| Power BI | Executive dashboards (separate) |

## App Pages

- **Project Overview** — KPIs, project flow, key finding
- **Churn Analysis** — Distribution, risk tiers, recency vs churn scatter
- **CLV & Revenue Risk** — CLV by tier, top 20 at-risk customers, risk scatter
- **Strategy Simulator** — Adjustable parameters, ROI comparison, cost vs saved
- **Customer Lookup** — Live churn probability gauge using trained model

## Dataset

UCI Online Retail II — publicly available at [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

- Period: Dec 2009 – Dec 2011
- Records: 1,067,371 transactions
- Customers: 4,967 (after cleaning)
- Market: UK-based online retailer

---

## 👤 Author
**Muhammed Nafih**  
Data Analyst | BI Developer

🔗 **LinkedIn:**  
https://www.linkedin.com/in/nafihhmohd/

---
