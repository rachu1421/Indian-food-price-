from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Wholesale vs Retail Prices", layout="wide")
sns.set_style("whitegrid")


def read_csv(file):
    return pd.read_csv(file, parse_dates=["date"])


@st.cache_resource
def train_model(d):
    X = pd.get_dummies(d[["category"]]).astype(float)
    X["log_w"] = np.log(d["wholesale"])
    X["days_back"] = d["days_back"]
    m = LinearRegression().fit(X, np.log(d["retail"]))
    return m, list(X.columns)


csv_path = Path(__file__).parent / "final_price_analysis_data.csv"

if csv_path.exists():
    df = read_csv(csv_path)
else:
    st.error("final_price_analysis_data.csv was not found next to app.py.")
    st.write("Files found in this folder:", sorted(p.name for p in csv_path.parent.iterdir()))
    st.write("Upload the csv to the same GitHub folder as app.py, or upload it below to use the app now.")
    uploaded = st.file_uploader("final_price_analysis_data.csv", type="csv")
    if uploaded is None:
        st.stop()
    df = read_csv(uploaded)

today = df[df.period == "today"].copy()

st.title("All India Wholesale vs Retail Prices")
st.caption("Source: Department of Consumer Affairs, report dated " + str(df["date"].max().date()))

page = st.sidebar.radio("Go to", ["Data", "EDA", "Spread and Margin", "Price Change",
                                  "Transmission", "Predict Retail Price", "Download"])

# ---------------------------------------------------------------- Data
if page == "Data":
    c1, c2, c3 = st.columns(3)
    c1.metric("Commodities", df["commodity"].nunique())
    c2.metric("Rows", len(df))
    c3.metric("Categories", df["category"].nunique())

    st.subheader("Cleaned data")
    cat = st.multiselect("Filter category", sorted(df["category"].unique()))
    show = df[df["category"].isin(cat)] if cat else df
    st.dataframe(show)

# ---------------------------------------------------------------- EDA
elif page == "EDA":
    st.subheader("Summary (today)")
    st.dataframe(today[["wholesale", "retail", "spread", "ratio", "margin_pct"]].describe().round(2))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(today["retail"], bins=12, ax=ax, color="darkorange")
        ax.set_title("Retail price distribution")
        st.pyplot(fig)
    with col2:
        order = today.groupby("category")["margin_pct"].mean().sort_values(ascending=False).index
        fig, ax = plt.subplots()
        sns.boxplot(data=today, x="category", y="margin_pct", order=order, ax=ax)
        ax.set_title("Margin % by category")
        st.pyplot(fig)

    pr = df.pivot(index="commodity", columns="period", values="retail")
    chg = pd.DataFrame({h: (pr["today"] / pr[h] - 1) * 100 for h in ["week_1", "month_1", "month_6", "year_1"]})
    vol = chg.std(axis=1).sort_values()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 7))
        vol.plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Volatility (std of % change)")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots()
        sns.heatmap(today[["wholesale", "retail", "spread", "margin_pct"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation")
        st.pyplot(fig)

# ---------------------------------------------------------------- Spread
elif page == "Spread and Margin":
    st.subheader("Retail minus wholesale (today)")
    t = today.sort_values("margin_pct", ascending=False)
    st.dataframe(t[["commodity", "category", "wholesale", "retail", "spread", "ratio", "margin_pct", "volatility"]].round(2))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(t["commodity"][::-1], t["margin_pct"][::-1], color="steelblue")
    ax.set_xlabel("margin %")
    st.pyplot(fig)

# ---------------------------------------------------------------- Change
elif page == "Price Change":
    pr = df.pivot(index="commodity", columns="period", values="retail")
    pw = df.pivot(index="commodity", columns="period", values="wholesale")
    hz = ["week_1", "month_1", "month_6", "year_1"]
    chg_r = pd.DataFrame({h: (pr["today"] / pr[h] - 1) * 100 for h in hz})

    st.subheader("Retail % change vs earlier dates")
    st.dataframe(chg_r.round(2).sort_values("year_1", ascending=False))

    item = st.selectbox("Pick a commodity", sorted(df["commodity"].unique()))
    s = df[df.commodity == item].sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s["date"], s["wholesale"], marker="o", label="wholesale")
    ax.plot(s["date"], s["retail"], marker="o", label="retail")
    ax.set_title(item)
    ax.set_ylabel("Rs per kg")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------- Transmission
elif page == "Transmission":
    st.subheader("Wholesale to retail")
    res = stats.linregress(today["wholesale"], today["retail"])
    res2 = stats.linregress(np.log(today["wholesale"]), np.log(today["retail"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Slope (retail per wholesale)", round(res.slope, 3))
    c2.metric("R2", round(res.rvalue ** 2, 3))
    c3.metric("Elasticity (log-log)", round(res2.slope, 3))
    st.write(f"retail = {res.intercept:.2f} + {res.slope:.3f} x wholesale")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=today, x="wholesale", y="retail", ax=ax)
    st.pyplot(fig)

    pw = df.pivot(index="commodity", columns="period", values="wholesale")
    pr = df.pivot(index="commodity", columns="period", values="retail")
    rows = []
    for h in ["week_1", "month_1", "month_6", "year_1"]:
        x = (pw["today"] / pw[h] - 1) * 100
        y = (pr["today"] / pr[h] - 1) * 100
        ok = x.notna() & y.notna()
        r = stats.linregress(x[ok], y[ok])
        rows.append([h, int(ok.sum()), r.slope, r.rvalue ** 2])
    st.write("Pass through of % changes (slope near 1 means retail follows wholesale fully)")
    st.dataframe(pd.DataFrame(rows, columns=["horizon", "n", "pass_through", "r2"]).round(3))

# ---------------------------------------------------------------- Predict
elif page == "Predict Retail Price":
    st.subheader("Predict retail price from wholesale price")
    st.caption("Linear regression on log prices. It estimates retail at the same time as the wholesale price, it is not a future forecast.")

    model, cols = train_model(df)
    item = st.selectbox("Commodity", sorted(today["commodity"].unique()))
    row = today[today.commodity == item].iloc[0]

    w = st.number_input("Wholesale price (Rs per kg)", min_value=1.0, value=float(round(row["wholesale"], 2)), step=0.5)

    x = pd.DataFrame(0.0, index=[0], columns=cols)
    x["log_w"] = np.log(w)
    x["days_back"] = 0
    x["category_" + row["category"]] = 1.0
    pred = float(np.exp(model.predict(x)[0]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted retail", f"Rs {pred:.2f}")
    c2.metric("Actual retail today", f"Rs {row['retail']:.2f}")
    c3.metric("Predicted margin", f"{(pred / w - 1) * 100:.1f} %")

    st.write("Volatility group of this commodity:", row["volatility"])
    st.info("Predictions for Onion and other items with unusual margins can be off by about 5 to 10 percent.")

    with st.expander("Model check: predicted vs actual retail (today, all commodities)"):
        chk = today[["commodity", "retail", "predicted_retail", "prediction_error_pct"]].sort_values("prediction_error_pct")
        st.dataframe(chk)

# ---------------------------------------------------------------- Download
else:
    st.subheader("Final dataset")
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="final_price_analysis_data.csv", mime="text/csv")