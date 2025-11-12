import streamlit as st
import pandas as pd
import joblib, json


MODEL = "artifacts/model.joblib"
META = "artifacts/metadata.json"

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧭",
    layout="wide"
)

st.image("data/logo.png", width=140)

st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        h1, h2, h3, h4 {
            color: #00C9A7;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            color: #00C9A7;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🧭 Employee Attrition Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#A9A9A9;'>Enter employee information to predict attrition probability.</p>", unsafe_allow_html=True)
st.markdown("---")


model = joblib.load(MODEL)
with open(META,'r') as f:
    meta = json.load(f)

input_data = {}

# ---------------- CATEGORICAL FIELDS (Dropdowns) ----------------
st.subheader("Categorical Attributes")

col1, col2 = st.columns(2)

with col1:
    input_data["BusinessTravel"] = st.selectbox("BusinessTravel",
        ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])

    input_data["Department"] = st.selectbox("Department",
        ["Human Resources", "Research & Development", "Sales"])

    input_data["EducationField"] = st.selectbox("EducationField",
        ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])

    input_data["Gender"] = st.selectbox("Gender", ["Female", "Male"])

    input_data["JobRole"] = st.selectbox("JobRole",
        ["Healthcare Representative", "Human Resources", "Laboratory Technician",
         "Manager", "Manufacturing Director", "Research Director", "Research Scientist",
         "Sales Executive", "Sales Representative"])

with col2:
    input_data["MaritalStatus"] = st.selectbox("MaritalStatus",
        ["Divorced", "Married", "Single"])

    input_data["OverTime"] = st.selectbox("OverTime", ["No", "Yes"])

    input_data["Education"] = st.selectbox("Education (1 = Below College → 5 = Doctorate)", [1,2,3,4,5])

    input_data["JobInvolvement"] = st.selectbox("JobInvolvement", [1,2,3,4])

    input_data["JobLevel"] = st.selectbox("JobLevel", [1,2,3,4,5])

    input_data["JobSatisfaction"] = st.selectbox("JobSatisfaction", [1,2,3,4])

    input_data["EnvironmentSatisfaction"] = st.selectbox("EnvironmentSatisfaction", [1,2,3,4])

    input_data["RelationshipSatisfaction"] = st.selectbox("RelationshipSatisfaction", [1,2,3,4])

    input_data["PerformanceRating"] = st.selectbox("PerformanceRating", [3,4])

    input_data["StockOptionLevel"] = st.selectbox("StockOptionLevel", [0,1,2,3])

    input_data["TrainingTimesLastYear"] = st.selectbox("TrainingTimesLastYear", [0,1,2,3,4,5,6])

    input_data["WorkLifeBalance"] = st.selectbox("WorkLifeBalance", [1,2,3,4])

    input_data["NumCompaniesWorked"] = st.selectbox("NumCompaniesWorked", list(range(0,10)))

# ---------------- NUMERIC FIELDS (Range Sliders) ----------------
st.subheader("Numeric Attributes")

input_data["Age"] = st.slider("Age", 18, 60, 30)
input_data["DailyRate"] = st.slider("DailyRate", 102, 1499, 800)
input_data["DistanceFromHome"] = st.slider("DistanceFromHome", 1, 29, 10)
input_data["HourlyRate"] = st.slider("HourlyRate", 30, 100, 60)
input_data["MonthlyIncome"] = st.slider("MonthlyIncome", 1009, 19999, 5000, step=100)
input_data["MonthlyRate"] = st.slider("MonthlyRate", 2094, 26999, 8000, step=100)
input_data["PercentSalaryHike"] = st.slider("PercentSalaryHike", 11, 25, 15)
input_data["TotalWorkingYears"] = st.slider("TotalWorkingYears", 0, 40, 10)
input_data["YearsAtCompany"] = st.slider("YearsAtCompany", 0, 40, 5)
input_data["YearsInCurrentRole"] = st.slider("YearsInCurrentRole", 0, 18, 5)
input_data["YearsSinceLastPromotion"] = st.slider("YearsSinceLastPromotion", 0, 15, 2)
input_data["YearsWithCurrManager"] = st.slider("YearsWithCurrManager", 0, 17, 5)

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict"):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[:,1][0]

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#1A1D23; border-radius:10px;'>
    """, unsafe_allow_html=True)

    if prob >= 0.5:
        st.markdown(f"<h2 style='color:#FF6B6B;'>🚨 Attrition Likely ({prob:.1%})</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#00C9A7;'>✅ Employee Likely to Stay ({prob:.1%})</h2>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr style='border:1px solid #00C9A7;'>
<p style='text-align:center; color:#A9A9A9; font-size:13px;'>
Built with ❤️ using Streamlit | © 2025 SharonGadi
</p>
""", unsafe_allow_html=True)
