import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Page Configuration
st.set_page_config(
    page_title="F1 Champion Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
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
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "f1_processed_data.csv")
        if not os.path.exists(csv_path):
            st.error(f"❌ File not found at: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "f1_champion_model.pkl")
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

# 4. Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.title("F1 Analytics Hub")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Analytics Dashboard",
    "🔮 Prediction Page",
    "🆚 Driver Comparison",
    "🧪 What-If Simulator",
    "🛠️ Team Impact Analysis", # <-- NEW PAGE HERE
    "ℹ️ About"
])

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.title("🏁 F1 World Champion Predictor")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Welcome to the Future of F1 Analytics! 🏎️
        This project uses AI to predict the F1 World Champion.
        
        **New Feature:**
        * 🛠️ **Team Impact Analysis:** See how Constructor points affect a driver's chances!
        """)
        st.info("👈 **Select a page from the Sidebar to start!**")
    with col2:
        st.image("https://media.formula1.com/image/upload/content/dam/fom-website/manual/Misc/2021-Master-Folder/F1%20logo.png", width=300)

# --- PAGE 2: ANALYTICS ---
elif page == "📊 Analytics Dashboard":
    st.title("📊 Historical Analytics")
    tab1, tab2 = st.tabs(["🏆 Champion Trends", "📈 Feature Correlations"])
    with tab1:
        champions = df[df['is_champion'] == 1]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=champions, x='season', y='win_rate', palette='magma', ax=ax)
        plt.xticks(rotation=45)
        plt.title("Win Rate of Champions")
        st.pyplot(fig)
    with tab2:
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_df[['points_per_race', 'win_rate', 'wins', 'position', 'is_champion']].corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

# --- PAGE 3: PREDICTION ---
# --- PAGE 3: PREDICTION PAGE (SMART FIX 🧠) ---
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
        
        # Get Probability for Selected Driver
        prob = model.predict_proba(features)[0][1]
        
        # --- SMART LOGIC: Compare with ALL drivers in 2025 ---
        # "Ivaru probability kammiya irundhalum, mathavangala vida mass ah?" nu paakrom.
        
        all_probs = {}
        for d in driver_list:
            d_stats = season_2025[season_2025['driver_name'] == d].iloc[0]
            d_feats = pd.DataFrame({
                'points_per_race': [d_stats['points_per_race']], 
                'win_rate': [d_stats['win_rate']], 
                'wins': [d_stats['wins']]
            })
            p = model.predict_proba(d_feats)[0][1]
            all_probs[d] = p
            
        # Find who has the HIGHEST probability in 2025
        best_driver = max(all_probs, key=all_probs.get)
        best_prob = all_probs[best_driver]
        
        # --- DISPLAY RESULTS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Wins", driver_stats['wins'])
        col2.metric("Points", driver_stats['points'])
        col3.metric("Win Rate", f"{driver_stats['win_rate']*100:.1f}%")
        
        st.divider()
        st.subheader("Prediction Result")
        
        # Progress bar
        st.progress(prob)
        st.metric(label="Championship Probability", value=f"{prob*100:.2f}%")
        
        # DECISION TIME 🏆
        if selected_driver == best_driver:
            st.success(f"🏆 **YES!** {selected_driver} is the **Statistical Favorite** to win the 2025 Championship!")
            st.markdown(f"*(Even though the probability is tight, he is ranked #1 among all drivers)*")
            st.balloons()
        elif prob > 0.4:
            st.success(f"🔥 **Strong Contender!** {selected_driver} has a great chance.")
        else:
            st.error(f"❌ **Unlikely.** The model predicts {best_driver} has a better chance ({best_prob*100:.1f}%).")
            
        # Optional: Show Top 3 Contenders
        with st.expander("👀 View Top 3 Contenders (Why is it tight?)"):
            st.write("This season is very competitive! Here are the top probabilities:")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for rank, (d_name, d_prob) in enumerate(sorted_probs, 1):
                st.write(f"**#{rank} {d_name}**: {d_prob*100:.2f}%")

# --- PAGE 4: COMPARISON ---
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

# --- PAGE 5: WHAT-IF ---
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

# --- PAGE 6: TEAM IMPACT ANALYSIS (NEW) ---
elif page == "🛠️ Team Impact Analysis":
    st.title("🛠️ Team & Constructor Impact")
    st.markdown("Analyze how **Team Performance (Constructor Points)** affects a driver's championship chances.")
    
    # 1. Select Driver
    season_2025 = df[df['season'] == 2025]
    driver_list = season_2025['driver_name'].unique()
    selected_driver = st.selectbox("Select Driver", driver_list)
    
    # Get current stats
    current_stats = season_2025[season_2025['driver_name'] == selected_driver].iloc[0]
    team_name = current_stats['constructor']
    
    # Calculate Team Stats (Approximate from data)
    team_drivers = season_2025[season_2025['constructor'] == team_name]
    current_team_points = team_drivers['points'].sum()
    current_driver_share = (current_stats['points'] / current_team_points) if current_team_points > 0 else 0
    
    st.info(f"🏎️ **Current Team:** {team_name} | **Total Team Points:** {current_team_points}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input: Constructor Points
        new_team_points = st.number_input("Projected Team Points", 
                                          min_value=0, max_value=1000, 
                                          value=int(current_team_points))
        
        # Input: Driver Share
        share_pct = st.slider("Driver's Share of Points (%)", 
                              min_value=0, max_value=100, 
                              value=int(current_driver_share*100))
        
    with col2:
        # Derived Driver Points
        derived_points = new_team_points * (share_pct / 100)
        st.metric("Projected Driver Points", f"{derived_points:.1f}")
        
        # Input: Wins
        wins = st.slider("Projected Driver Wins", 0, 24, int(current_stats['wins']))
        
    # Prediction Logic
    if st.button("Analyze Team Impact"):
        total_races = 24 # 2025 Season
        
        # Calculate Features for Model
        new_ppr = derived_points / total_races
        new_win_rate = wins / total_races
        
        features = pd.DataFrame({
            'points_per_race': [new_ppr],
            'win_rate': [new_win_rate],
            'wins': [wins]
        })
        
        # Predict
        prob = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        st.divider()
        st.subheader("📢 Impact Result")
        
        st.metric(label="New Championship Probability", value=f"{prob*100:.2f}%")
        
        if prediction == 1 or prob > 0.4:
            st.success(f"🚀 With these team stats, {selected_driver} **CAN WIN** the Championship!")
            st.balloons()
        else:
            st.warning(f"⚠️ Even with these team stats, it's difficult for {selected_driver} to win.")
            st.markdown("*Try increasing the Driver Share % or Wins!*")

# --- PAGE 7: ABOUT ---
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### 🎯 Project Goal
    To predict the Formula 1 World Champion using Machine Learning.
    
    **Developed by:** *TOTZ* 😉
    """)
