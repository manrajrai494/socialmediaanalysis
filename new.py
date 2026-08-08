import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Social Media & Mental Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")

def set_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
          .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}")!important;
                background-size: cover!important;
                background-attachment: fixed!important;
            }}
            [data-testid="stAppViewContainer"] >.main {{
                background-color: rgba(255, 255, 255, 0.88)!important;
                padding: 25px; border-radius: 15px;
            }}
            [data-testid="stSidebar"] {{
                background-color: rgba(255, 255, 255, 0.92)!important;
            }}
            </style>
            """, unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found.")

set_bg_from_local("satinder.gif")

# ----------------------------
# Load Dataset + ADD STATE + FIX PARSER ERROR
# ----------------------------
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
    'Uttarakhand', 'West Bengal'
]

@st.cache_data
def load_data():
    # FIX: on_bad_lines='skip' and engine='python' to handle broken rows
    df = pd.read_csv(
        "social_media.csv.csv",
        on_bad_lines='skip',
        engine='python',
        skip_blank_lines=True
    )

    # Remove duplicate header rows if pasted twice
    df = df[df['User_ID']!= 'User_ID']

    df.columns = df.columns.str.strip()
    df.replace("", pd.NA, inplace=True)

    numeric_cols = [
        "Age", "Daily_Usage_Hours", "Anxiety_Score", "Depression_Score",
        "Sleep_Quality_Score", "Sleep_Hours", "Mental_Wellbeing_Score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["Gender", "Occupation", "Primary_Platform", "Tried_To_Cut_Back", "Addiction_Level"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ADD STATE COLUMN FOR MAP
    if 'India_State' not in df.columns:
        np.random.seed(42)
        df['India_State'] = np.random.choice(INDIAN_STATES, size=len(df))

    if "Daily_Usage_Hours" in df.columns:
        df["Usage_Category"] = pd.cut(df["Daily_Usage_Hours"], bins=[0,2,4,24], labels=["Low", "Medium", "High"])

    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🧠 Mental Health Analytics")
st.sidebar.markdown("MCA Final Year Project")

st.sidebar.header("🎯 Filters")
gender = st.sidebar.multiselect("Gender", options=sorted(df["Gender"].dropna().unique()), default=sorted(df["Gender"].dropna().unique()))
platform = st.sidebar.multiselect("Platform", options=sorted(df["Primary_Platform"].dropna().unique()), default=sorted(df["Primary_Platform"].dropna().unique()))
occupation = st.sidebar.multiselect("Occupation", options=sorted(df["Occupation"].dropna().unique()), default=sorted(df["Occupation"].dropna().unique()))
addiction = st.sidebar.multiselect("Addiction Level", options=sorted(df["Addiction_Level"].dropna().unique()), default=sorted(df["Addiction_Level"].dropna().unique()))
state_filter = st.sidebar.multiselect("State", options=sorted(df["India_State"].unique()), default=sorted(df["India_State"].unique()))
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

# Apply Filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]
filtered_df = filtered_df[filtered_df["Occupation"].isin(occupation)]
filtered_df = filtered_df[filtered_df["Primary_Platform"].isin(platform)]
filtered_df = filtered_df[filtered_df["Addiction_Level"].isin(addiction)]
filtered_df = filtered_df[filtered_df["India_State"].isin(state_filter)]
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Overview", "📊 EDA", "📈 Insights", "🔥 Correlation", "💾 Download", "🗺️ State Map"])

with tab1:
    st.title("📊 Social Media & Mental Health Dashboard")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Respondents", len(filtered_df))
    k2.metric("Avg Daily Usage", f"{filtered_df['Daily_Usage_Hours'].mean():.2f} hrs")
    k3.metric("Avg Anxiety Score", f"{filtered_df['Anxiety_Score'].mean():.2f}")
    k4.metric("Avg Wellbeing Score", f"{filtered_df['Mental_Wellbeing_Score'].mean():.2f}")
    st.dataframe(filtered_df.head(15), use_container_width=True)

with tab2:
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        filtered_df["Gender"].value_counts().plot(kind="bar", color=sns.color_palette("pastel"), ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        filtered_df["Addiction_Level"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

with tab3:
    st.header("📈 Key Insights")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Daily_Usage_Hours", y="Anxiety_Score", hue="Addiction_Level", ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("🔥 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

with tab5:
    st.header("💾 Download Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Filtered Data", data=csv, file_name='filtered_social_media_data.csv', mime='text/csv')

with tab6:
    st.title("🗺️ State Wise User Distribution")
    state_counts = filtered_df['India_State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    state_latlong = {
        'Andhra Pradesh': [15.9129, 79.7400], 'Arunachal Pradesh': [28.2180, 94.7278], 'Assam': [26.2006, 92.9376],
        'Bihar': [25.0961, 85.3131], 'Chhattisgarh': [21.2787, 81.8661], 'Goa': [15.2993, 74.1240],
        'Gujarat': [22.2587, 71.1924], 'Haryana': [29.0588, 76.0856], 'Himachal Pradesh': [31.1048, 77.1734],
        'Jharkhand': [23.6102, 85.2799], 'Karnataka': [15.3173, 75.7139], 'Kerala': [10.8505, 76.2711],
        'Madhya Pradesh': [22.9734, 78.6569], 'Maharashtra': [19.7515, 75.7139], 'Manipur': [24.6637, 93.9063],
        'Meghalaya': [25.4670, 91.3662], 'Mizoram': [23.1645, 92.9376], 'Nagaland': [26.1584, 94.5624],
        'Odisha': [20.9517, 85.0985], 'Punjab': [31.1471, 75.3412], 'Rajasthan': [27.0238, 74.2179],
        'Sikkim': [27.5330, 88.5122], 'Tamil Nadu': [11.1271, 78.6569], 'Telangana': [18.1124, 79.0193],
        'Tripura': [23.9408, 91.9882], 'Uttar Pradesh': [26.8467, 80.9462], 'Uttarakhand': [30.0668, 79.0193],
        'West Bengal': [22.9868, 87.8550]
    }
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
    for idx, row in state_counts.iterrows():
        state = row['State']
        count = row['Count']
        if state in state_latlong:
            lat, lon = state_latlong[state]
            folium.CircleMarker(location=[lat, lon], radius=count/10 + 5, popup=f"<b>{state}</b><br>Users: {count}", color='blue', fill=True).add_to(m)
    st_folium(m, width=700, height=500)

st.caption("© 2026 MCA Project | Social Media Mental Health Analytics")