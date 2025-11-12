import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.image("data/logo.png", width=140)
st.markdown("<h1 style='text-align:center; color:#00C9A7;'>📈 Attrition Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#A9A9A9;'>Explore key employee trends and attrition patterns.</p>", unsafe_allow_html=True)

df = pd.read_csv("data/greendestination.csv")

col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df))
col2.metric("Attrition Rate", f"{df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")
col3.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

default_feature = "Department" if "Department" in df.columns else df.columns[0]
feature = st.selectbox("Select a feature to compare with Attrition:", df.columns, index=df.columns.get_loc(default_feature))


fig = px.histogram(df, x=feature, color="Attrition", barmode="group",
                   color_discrete_map={"Yes": "#EF553B", "No": "#00C9A7"},
                   template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)


st.markdown("""
<hr style='border:1px solid #00C9A7;'>
<p style='text-align:center; color:#A9A9A9; font-size:13px;'>
Built with ❤️ using Streamlit | © 2025 SharonGadi
</p>
""", unsafe_allow_html=True)
