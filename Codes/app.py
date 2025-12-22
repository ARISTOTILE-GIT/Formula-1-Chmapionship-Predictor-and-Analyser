import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="F1 Champion Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff1801;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Data & Model
@st.cache_data
def load_data():
    df = pd.read_csv("f1_processed_data.csv")
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load("f1_champion_model.pkl")
        return model
    except:
        return None

df = load_data()
model = load_model()

if df is None:
    st.error("⚠️ Data file 'f1_processed_data.csv' not found.")
    st.stop()

if model is None:
    st.error("⚠️ Model file 'f1_champion_model.pkl' not found. Run step3_final_train.py first!")
    st.stop()

# 4. Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.title("F1 Analytics Hub")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Analytics Dashboard",
    "🔮 Prediction Page",
    "🆚 Driver Comparison",
    "🧪 What-If Simulator",
    "ℹ️ About"
])

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.title("🏁 F1 World Champion Predictor")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Welcome to the Future of F1 Analytics! 🏎️
        
        This project leverages **Machine Learning** to predict the probability of a driver becoming the World Champion. 
        We analyze historical data (2000-2025) to understand what it takes to win.
        
        **Key Features:**
        * 📊 **Deep Analytics:** Explore historical trends.
        * 🔮 **AI Predictions:** Real-time probability for 2025 drivers.
        * 🆚 **Head-to-Head:** Compare Lando Norris vs Max Verstappen.
        * 🧪 **Simulator:** Change wins/points and see the magic.
        """)
        st.info("👈 **Start by selecting a page from the Sidebar!**")
        
    with col2:
        st.image("https://media.formula1.com/image/upload/content/dam/fom-website/manual/Misc/2021-Master-Folder/F1%20logo.png", width=300)

# --- PAGE 2: ANALYTICS DASHBOARD ---
elif page == "📊 Analytics Dashboard":
    st.title("📊 Historical Analytics")
    
    tab1, tab2 = st.tabs(["🏆 Champion Trends", "📈 Feature Correlations"])
    
    with tab1:
        st.subheader("What does it take to be a Champion?")
        champions = df[df['is_champion'] == 1]
        
        # Plot 1: Win Rate of Champions
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=champions, x='season', y='win_rate', palette='magma', ax=ax)
        plt.xticks(rotation=45)
        plt.title("Win Rate of Every Champion (2000-2025)")
        plt.ylabel("Win Rate (0.0 - 1.0)")
        st.pyplot(fig)
        
        st.markdown("**Observation:** Notice how some years (e.g., 2004, 2023) have huge bars? That's dominance. Lower bars (e.g., 2008) mean a tight fight!")

    with tab2:
        st.subheader("Correlation Heatmap")
        st.write("Which stats matter most?")
        
        corr_matrix = df[['points_per_race', 'win_rate', 'wins', 'position', 'is_champion']].corr()
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)
        
        st.write("Higher correlation with `is_champion` means that feature is very important!")

# --- PAGE 3: PREDICTION PAGE ---
elif page == "🔮 Prediction Page":
    st.title("🔮 2025 Championship Predictor")
    
    st.subheader("Select a Driver from the 2025 Grid")
    
    season_2025 = df[df['season'] == 2025]
    driver_list = season_2025['driver_name'].unique()
    selected_driver = st.selectbox("Choose Driver", driver_list, index=0)
    
    if st.button("Predict Probability"):
        driver_stats = season_2025[season_2025['driver_name'] == selected_driver].iloc[0]
        
        # Features for model
        features = pd.DataFrame({
            'points_per_race': [driver_stats['points_per_race']],
            'win_rate': [driver_stats['win_rate']],
            'wins': [driver_stats['wins']]
        })
        
        # Prediction
        prob = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Wins", driver_stats['wins'])
        col2.metric("Points", driver_stats['points'])
        col3.metric("Win Rate", f"{driver_stats['win_rate']*100:.1f}%")
        
        st.divider()
        st.subheader("Prediction Result")
        
        my_bar = st.progress(0)
        for percent_complete in range(int(prob*100)):
            my_bar.progress(percent_complete + 1)
            
        st.metric(label="Championship Probability", value=f"{prob*100:.2f}%")
        
        if prob > 0.5:
            st.success(f"🏆 **Highly Likely to be Champion!**")
            st.balloons()
        else:
            st.error("❌ **Unlikely to win.**")

# --- PAGE 4: DRIVER COMPARISON ---
elif page == "🆚 Driver Comparison":
    st.title("🆚 Head-to-Head Comparison")
    
    season_2025 = df[df['season'] == 2025]
    drivers = season_2025['driver_name'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        driver1 = st.selectbox("Select Driver A", drivers, index=0) # Lando
    with col2:
        driver2 = st.selectbox("Select Driver B", drivers, index=1) # Max
        
    if st.button("Compare Drivers"):
        # Get Data
        d1_stats = season_2025[season_2025['driver_name'] == driver1].iloc[0]
        d2_stats = season_2025[season_2025['driver_name'] == driver2].iloc[0]
        
        # Prepare Features
        f1 = pd.DataFrame({'points_per_race':[d1_stats['points_per_race']], 'win_rate':[d1_stats['win_rate']], 'wins':[d1_stats['wins']]})
        f2 = pd.DataFrame({'points_per_race':[d2_stats['points_per_race']], 'win_rate':[d2_stats['win_rate']], 'wins':[d2_stats['wins']]})
        
        # Get Probabilities
        prob1 = model.predict_proba(f1)[0][1]
        prob2 = model.predict_proba(f2)[0][1]
        
        # Display Side-by-Side
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader(driver1)
            st.write(f"**Wins:** {d1_stats['wins']}")
            st.write(f"**Points:** {d1_stats['points']}")
            st.info(f"🏆 Probability: **{prob1*100:.2f}%**")
            
        with c2:
            st.subheader(driver2)
            st.write(f"**Wins:** {d2_stats['wins']}")
            st.write(f"**Points:** {d2_stats['points']}")
            st.info(f"🏆 Probability: **{prob2*100:.2f}%**")
            
        st.divider()
        if prob1 > prob2:
            st.success(f"🥇 **{driver1}** has a higher chance of winning!")
        elif prob2 > prob1:
            st.success(f"🥇 **{driver2}** has a higher chance of winning!")
        else:
            st.warning("It's a tie!")

# --- PAGE 5: WHAT-IF SIMULATOR ---
elif page == "🧪 What-If Simulator":
    st.title("🧪 What-If Simulator")
    st.markdown("Adjust the stats below to see how **Championship Probability** changes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sim_wins = st.slider("Number of Wins", 0, 24, 8)
        sim_races = st.slider("Total Races", 10, 24, 24)
        
    with col2:
        sim_points = st.number_input("Total Points", 0, 600, 300)
        
    # Calculate derived metrics
    sim_win_rate = sim_wins / sim_races
    sim_ppr = sim_points / sim_races
    
    st.write(f"📊 **Calculated Stats:** Win Rate: `{sim_win_rate*100:.1f}%` | Points/Race: `{sim_ppr:.1f}`")
    
    # Predict
    sim_features = pd.DataFrame({
        'points_per_race': [sim_ppr],
        'win_rate': [sim_win_rate],
        'wins': [sim_wins]
    })
    
    sim_prob = model.predict_proba(sim_features)[0][1]
    
    # Gauge Chart (Simple Progress Bar for now)
    st.subheader("Champion Probability")
    st.progress(sim_prob)
    st.metric(label="Probability", value=f"{sim_prob*100:.2f}%")
    
    if sim_prob > 0.8:
        st.success("🔥 This performance is **Legendary**!")
    elif sim_prob > 0.5:
        st.info("👍 Good chance of winning.")
    else:
        st.error("👎 Not enough to win the title.")

# --- PAGE 6: ABOUT ---
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    
    st.markdown("""
    ### 🎯 Project Goal
    To predict the Formula 1 World Champion using Machine Learning, analyzing patterns from historical data (2000-2025).
    
    ### 📂 Dataset
    * **Source:** Ergast Developer API & FastF1 Library.
    * **Range:** 2000 to 2025 Season.
    * **Key Features:** Wins, Points, Win Rate, Consistency.
    
    ### 🤖 Model Used
    * **Algorithm:** Random Forest Classifier.
    * **Why?** It handles non-linear relationships well (e.g., high points but low wins) and prevents overfitting.
    * **Accuracy:** ~98% on test data.
    
    ### 👨‍💻 Tech Stack
    * **Python** (Logic)
    * **Pandas** (Data Manipulation)
    * **Scikit-Learn** (Machine Learning)
    * **Streamlit** (Web App Interface)
    
    **Developed by:** *Un Machi* 😉
    """)
