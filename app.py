import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score

# =====================================================================
# 💻 STYLE & THEME CONFIGURATION
# =====================================================================
st.set_page_config(page_title="BoxOfficeBrain Engine", page_icon="🧠", layout="wide")

# Professional "Netflix-Style" dark theme injection
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #FFFFFF; }
    h1 { color: #E50914 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: bold; }
    h2, h3 { color: #F5F5F1 !important; font-weight: bold; }
    .stButton>button { background-color: #E50914 !important; color: white !important; border-radius: 4px; border: none; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #B20710 !important; }
    div[data-testid="stMetricValue"] { color: #E50914 !important; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #1c1c1c; }
    .explain-box { background-color: #222222; padding: 15px; border-left: 4px solid #E50914; border-radius: 4px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 📊 DATA MANAGEMENT & PIPELINE
# =====================================================================
@st.cache_data
def load_and_clean_data():
    # 1. Generate programmatic raw historical dataset (200 movies)
    np.random.seed(42)
    genres_list = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']
    data = {
        'movie_id': range(1, 201),
        'title': [f"Historical Film {i}" for i in range(1, 201)],
        'budget_millions': np.random.randint(10, 250, size=200).astype(float),
        'runtime_minutes': np.random.randint(85, 175, size=200).astype(float),
        'genre': np.random.choice(genres_list, size=200),
        'user_rating': np.round(np.random.uniform(4.5, 9.2, size=200), 1)
    }
    
    # Intentionally inject missing values to demonstrate Cleaning rules
    df_raw = pd.DataFrame(data)
    df_raw.iloc[np.random.choice(200, 10, replace=False), 3] = np.nan 

    # 2. SQL Database Pipeline using SQLAlchemy Core
    engine = create_engine('sqlite:///boxoffice_movies.db')
    df_raw.to_sql('raw_table', engine, if_exists='replace', index=False)
    df = pd.read_sql_query("SELECT * FROM raw_table", engine)

    # 3. Data Cleansing (Handling missing data with .fillna)
    median_runtime = df['runtime_minutes'].median()
    df['runtime_minutes'] = df['runtime_minutes'].fillna(median_runtime)

    # 4. Feature Target Transformation
    df['revenue_millions'] = (df['budget_millions'] * np.random.uniform(0.6, 2.8, size=200)) + (df['user_rating'] * 3)
    df['is_hit'] = (df['user_rating'] >= 7.0).astype(int)
    
    return df

df = load_and_clean_data()

# One-hot encoding text values into math feature flags
df_encoded = pd.get_dummies(df, columns=['genre'], drop_first=True)
features = [col for col in df_encoded.columns if col not in ['movie_id', 'title', 'revenue_millions', 'is_hit', 'user_rating']]
X = df_encoded[features]

# =====================================================================
# 🤖 MACHINE LEARNING ENGINE
# =====================================================================
# Task 1: Regression Model (Predicting continuous dollar numbers)
y_reg = df_encoded['revenue_millions']
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_model = LinearRegression().fit(X_train, y_train_reg)
reg_preds = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
r2 = r2_score(y_test_reg, reg_preds)

# Task 2: Classification Model (Predicting discrete Yes/No hit states)
y_clf = df_encoded['is_hit']
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
clf_model = LogisticRegression().fit(X_train, y_train_clf)
clf_preds = clf_model.predict(X_test)
accuracy = accuracy_score(y_test_clf, clf_preds)
precision = precision_score(y_test_clf, clf_preds)
recall = recall_score(y_test_clf, clf_preds)

# =====================================================================
# 🌐 EXPLANATORY STREAMLIT UI DESIGN
# =====================================================================
st.markdown("<h1>🧠 BoxOfficeBrain</h1>", unsafe_allow_html=True)
st.markdown("### *An AI-Powered Script Viability & Revenue Prediction Engine*")
st.write("---")

# Left Sidebar: Presentation Checklist (Removed Unit titles)
st.sidebar.markdown("## 📋 Syllabus Mapping Checklist")
st.sidebar.info("""
**Project Introduction**
* App built using **Streamlit** micro-framework.
* Sets performance metric anchors.

**Data Clean & ETL**
* Loaded clean via SQL **SQLAlchemy Engine**.
* Missing values cleaned using **.fillna()**.
* Manipulations processed via **pandas DataFrames**.

**Scikit-Learn ML Tasks**
* **Regression:** Predicts exact movie revenue dollars.
* **Classification:** Predicts binary success (Hit/Flop).
* **Recommendation:** Uses **Cosine Similarity** arrays.
""")

st.sidebar.markdown("### 📊 Live Model Scorecard")
st.sidebar.markdown(f"**Regression Performance:**\n* **RMSE (Average Error):** `${rmse:.2f}M` \n* **R² (Accuracy Variance):** `{r2:.2f}`")
st.sidebar.markdown(f"**Classification Performance:**\n* **Total Accuracy:** `{accuracy*100:.1f}%` \n* **Precision Score:** `{precision:.2f}` \n* **Recall Score:** `{recall:.2f}`")

# Main Screen Split
col_input, col_viz = st.columns([1, 1.2])

with col_input:
    st.markdown("### 💡 Step 1: Input a Movie Concept")
    st.write("Adjust the parameters below to represent your new script idea.")
    input_title = st.text_input("Proposed Movie Title", "Project X-Files")
    input_budget = st.slider("Production Budget ($ Millions)", 10, 250, 65)
    input_runtime = st.slider("Target Runtime (Minutes)", 85, 175, 115)
    input_genre = st.selectbox("Primary Genre Focus", ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'])
    
    st.write("")
    submit_btn = st.button("🚀 EXECUTE MACHINE LEARNING PREDICTION")

with col_viz:
    st.markdown("### 📈 Step 2: Understand the Historical Baselines")
    st.write("This chart allows us to visually inspect how historical film collections are distributed across categories before model matching.")
    
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    
    sns.histplot(data=df, x='revenue_millions', hue='genre', multiple='stack', ax=ax, palette='autumn', edgecolor='#111111')
    ax.set_title("Historical Global Revenue by Genre Groupings", color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.set_xlabel("Revenue ($M)", color='white', fontsize=8)
    ax.set_ylabel("Count of Records", color='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# Processing the execution output report block cleanly
if submit_btn:
    st.write("---")
    st.markdown(f"## 📊 Assessment Output: {input_title}")
    
    # Structural normalization to fit mathematical shape models
    input_data = {'budget_millions': input_budget, 'runtime_minutes': input_runtime}
    for g in ['Comedy', 'Drama', 'Sci-Fi', 'Horror']:
        input_data[f'genre_{g}'] = 1.0 if input_genre == g else 0.0
    input_df = pd.DataFrame([input_data])[X.columns]
    
    # Generate direct numeric evaluation floats
    pred_revenue = float(reg_model.predict(input_df))
    pred_hit = int(clf_model.predict(input_df))
    
    card1, card2 = st.columns(2)
    with card1:
        st.markdown("<div class='explain-box'><b>What is this?</b> Our <b>Regression Model</b> uses historical trends to calculate the exact global box office earnings.</div>", unsafe_allow_html=True)
        st.metric(label="Estimated Global Box Office Revenue Prediction", value=f"${pred_revenue:.1f} Million")
    with card2:
        st.markdown("<div class='explain-box'><b>What is this?</b> Our <b>Classification Model</b> checks features to determine if user review ratings will cross above a 7.0 score.</div>", unsafe_allow_html=True)
        if pred_hit == 1:
            st.success("🎉 FORECAST: This movie layout is structurally built to be a Critical Review Hit!")
        else:
            st.warning("⚠️ RISK NOTICE: Mathematical paths expect lower user reviews (Rating < 7.0).")
            
    # Content-Based Recommendations Benchmark Segment
    st.write("")
    st.markdown("### 🍿 Step 3: View 3 Structurally Identical Benchmark Movies")
    st.write("Using a mathematical tool called **Cosine Similarity**, our system scans the database to find the 3 closest historical matches to your proposed movie characteristics so production executives can cross-compare.")
    
    # Flatten array elements properly to avoid vector slicing crashes
    input_sims = cosine_similarity(input_df, X).flatten()
    top_indices = np.argsort(input_sims)[-3:][::-1]
    
    rec_cols = st.columns(3)
    for idx, col in enumerate(rec_cols):
        match_idx = top_indices[idx]
        matched_movie = df.iloc[match_idx]
        with col:
            st.markdown(f"<div style='background-color:#222; padding:15px; border-radius:5px;'>", unsafe_allow_html=True)
            st.markdown(f"**🎬 Peer Match #{idx+1}: {matched_movie['title']}**")
            st.write(f"• **Genre:** {matched_movie['genre']}")
            st.write(f"• **Budget spent:** ${float(matched_movie['budget_millions']):.1f}M")
            st.write(f"• **Revenue made:** ${float(matched_movie['revenue_millions']):.1f}M")