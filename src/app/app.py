"""
Streamlit Frontend for World Cup 2026 Simulator
Complete interface for training, simulation, and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.tournament import TournamentSimulator
from src.simulation.predictor_integration import create_predictor_from_models


# Page configuration
st.set_page_config(
    page_title="FIFA World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application."""
    st.title("⚽ FIFA World Cup 2026 Simulator")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Home", "🎯 Run Simulation", "📊 View Results", "🔧 Model Training", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Run Simulation":
        show_simulation()
    elif page == "📊 View Results":
        show_results()
    elif page == "🔧 Model Training":
        show_training()
    elif page == "ℹ️ About":
        show_about()


def show_home():
    """Home page with overview."""
    st.header("Welcome to the World Cup 2026 Simulator!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Teams", "48")
        st.metric("Groups", "12 (A-L)")
    
    with col2:
        st.metric("Knockout Teams", "32")
        st.metric("Total Matches", "104")
    
    with col3:
        st.metric("Rounds", "6")
        st.metric("Champion Plays", "8 matches")
    
    st.markdown("---")
    
    st.subheader("Tournament Format")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Group Stage:**
        - 12 groups of 4 teams each
        - Round-robin within groups
        - Top 2 from each group qualify (24 teams)
        - Best 8 third-place teams also qualify
        - **Total: 32 teams advance**
        """)
    
    with col2:
        st.markdown("""
        **Knockout Stage:**
        - Round of 32 (32 → 16)
        - Round of 16 (16 → 8)
        - Quarter-finals (8 → 4)
        - Semi-finals (4 → 2)
        - Final (2 → 1)
        - **Extra time & penalties if needed**
        """)
    
    st.markdown("---")
    
    st.subheader("How It Works")
    
    st.markdown("""
    This simulator uses:
    1. **Custom ELO Ratings** - Competition-weighted rating system
    2. **Machine Learning Models** - XGBoost & CatBoost trained on historical data
    3. **Poisson Regression** - Statistical baseline for expected goals
    4. **Dixon-Coles Model** - Adjustment for low-scoring games
    5. **Monte Carlo Simulation** - Run thousands of tournaments to estimate probabilities
    
    Navigate to **🎯 Run Simulation** to get started!
    """)
    
    # Show groups
    st.markdown("---")
    st.subheader("Official 2026 World Cup Groups")
    
    from src.simulation.groups import WORLD_CUP_GROUPS
    
    cols = st.columns(4)
    
    for idx, (group_name, teams) in enumerate(sorted(WORLD_CUP_GROUPS.items())):
        col_idx = idx % 4
        with cols[col_idx]:
            st.markdown(f"**Group {group_name}**")
            for team in teams:
                st.write(f"• {team}")
            st.write("")


def show_simulation():
    """Simulation page."""
    st.header("🎯 Run Tournament Simulation")
    
    # Simulation settings
    st.subheader("Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=1,
            max_value=100000,
            value=10000,
            step=1000,
            help="More simulations = more accurate probabilities (but slower)"
        )
        
        use_dixon_coles = st.checkbox(
            "Use Dixon-Coles Model",
            value=True,
            help="Improves accuracy for low-scoring games"
        )
    
    with col2:
        base_model = st.selectbox(
            "Base Prediction Model",
            ["ensemble", "xgb", "cat", "poisson"],
            help="Which model to use for goal predictions"
        )
        
        detailed_mode = st.checkbox(
            "Detailed Mode (Single Simulation)",
            value=False,
            help="Show match-by-match results for 1 simulation"
        )
    
    st.markdown("---")
    
    # Run simulation button
    if detailed_mode:
        if st.button("🎮 Run Detailed Tournament Simulation", type="primary"):
            run_detailed_simulation(use_dixon_coles, base_model)
    else:
        if st.button(f"🚀 Run {n_simulations:,} Monte Carlo Simulations", type="primary"):
            run_monte_carlo(n_simulations, use_dixon_coles, base_model)


def run_detailed_simulation(use_dixon_coles: bool, base_model: str):
    """Run a single detailed tournament simulation."""
    st.info("Running detailed tournament simulation...")
    
    try:
        # Create predictor
        predictor = create_predictor_from_models(use_dixon_coles, base_model)
        
        # Create simulator
        simulator = TournamentSimulator(predictor)
        
        # Run simulation
        with st.spinner("Simulating tournament..."):
            result = simulator.simulate_tournament(detailed=True)
        
        # Display results
        st.success("✅ Simulation complete!")
        
        # Champion
        st.markdown(f"## 🏆 Champion: **{result['champion']}**")
        st.markdown("---")
        
        # Group stage
        st.subheader("📋 Group Stage Results")
        
        tabs = st.tabs([f"Group {g}" for g in sorted(result['group_standings'].keys())])
        
        for idx, group_name in enumerate(sorted(result['group_standings'].keys())):
            with tabs[idx]:
                standings = result['group_standings'][group_name]
                st.dataframe(
                    standings.style.highlight_max(subset=['points'], color='lightgreen'),
                    hide_index=True
                )
        
        # Qualified teams
        st.markdown("---")
        st.subheader("✅ Qualified Teams (32)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Group Winners (12)**")
            for i, team in enumerate(result['qualified_teams']['winners'], 1):
                st.write(f"{i}. {team}")
        
        with col2:
            st.markdown("**Runners-up (12)**")
            for i, team in enumerate(result['qualified_teams']['runners_up'], 1):
                st.write(f"{i}. {team}")
        
        with col3:
            st.markdown("**Best 3rd Place (8)**")
            for i, team in enumerate(result['qualified_teams']['third_place'], 1):
                st.write(f"{i}. {team}")
        
        # Knockout matches
        st.markdown("---")
        st.subheader("🏆 Knockout Stage")
        
        knockout_matches = result['knockout_matches']
        
        # Group by round
        rounds = {}
        for match in knockout_matches:
            if match.round_name not in rounds:
                rounds[match.round_name] = []
            rounds[match.round_name].append(match)
        
        round_order = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        
        for round_name in round_order:
            if round_name in rounds:
                with st.expander(f"**{round_name}**", expanded=(round_name == "Final")):
                    for match in rounds[round_name]:
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            st.write(f"**{match.home_team}**")
                        
                        with col2:
                            score_text = f"{match.home_goals_90} - {match.away_goals_90}"
                            if match.went_to_extra_time:
                                score_text += f" (AET: {match.home_goals_total}-{match.away_goals_total})"
                            if match.went_to_penalties:
                                score_text += " (Pens)"
                            st.write(score_text)
                        
                        with col3:
                            st.write(f"**{match.away_team}**")
                        
                        st.write(f"→ Winner: **{match.winner}**")
                        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error during simulation: {e}")
        import traceback
        st.code(traceback.format_exc())


def run_monte_carlo(n_simulations: int, use_dixon_coles: bool, base_model: str):
    """Run Monte Carlo simulation."""
    st.info(f"Running {n_simulations:,} tournament simulations...")
    
    try:
        # Create predictor
        with st.spinner("Loading prediction models..."):
            predictor = create_predictor_from_models(use_dixon_coles, base_model)
        
        # Create simulator
        simulator = TournamentSimulator(predictor)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Progress: {current:,}/{total:,} ({progress*100:.1f}%)")
        
        # Run simulation
        start_time = time.time()
        results_df = simulator.run_monte_carlo(n_simulations, progress_callback)
        elapsed = time.time() - start_time
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Completed {n_simulations:,} simulations in {elapsed:.1f}s!")
        
        # Save results
        results_df.to_csv('./results/monte_carlo_results.csv', index=False)
        
        # Display results
        st.markdown("---")
        st.subheader("🏆 Championship Probabilities")
        
        # Top 20 favorites
        top_20 = results_df.head(20)
        
        # Bar chart
        fig = px.bar(
            top_20,
            x='team',
            y='championship_probability',
            title='Top 20 Championship Favorites',
            labels={'championship_probability': 'Win Probability', 'team': 'Team'},
            color='championship_probability',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(
            top_20[['team', 'championship_probability', 'knockout_probability', 'avg_group_position']]
            .style.format({
                'championship_probability': '{:.2%}',
                'knockout_probability': '{:.2%}',
                'avg_group_position': '{:.2f}'
            }),
            hide_index=True
        )
        
        # Most likely champion
        champion = results_df.iloc[0]
        st.markdown(f"### 🥇 Most Likely Champion: **{champion['team']}** ({champion['championship_probability']*100:.2f}%)")
        
        # Save to session state
        st.session_state['last_results'] = results_df
        
    except Exception as e:
        st.error(f"Error during simulation: {e}")
        import traceback
        st.code(traceback.format_exc())


def show_results():
    """Results viewing page."""
    st.header("📊 Simulation Results")
    
    # Check for saved results
    if 'last_results' in st.session_state:
        results_df = st.session_state['last_results']
        
        st.success("Showing results from last simulation run")
        
        # Filter options
        st.subheader("Filter Results")
        
        min_prob = st.slider(
            "Minimum Championship Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            step=0.1
        )
        
        filtered_df = results_df[results_df['championship_probability'] >= min_prob/100]
        
        st.write(f"Showing {len(filtered_df)} teams with ≥{min_prob}% win probability")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Championship probability
            fig1 = px.pie(
                filtered_df.head(10),
                values='championship_probability',
                names='team',
                title='Top 10 - Championship Share'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Knockout probability
            fig2 = px.bar(
                filtered_df.head(10),
                x='team',
                y='knockout_probability',
                title='Top 10 - Knockout Qualification Probability',
                color='knockout_probability',
                color_continuous_scale='Blues'
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Full data table
        st.subheader("Complete Results")
        st.dataframe(
            filtered_df.style.format({
                'championship_probability': '{:.4%}',
                'knockout_probability': '{:.2%}',
                'avg_group_position': '{:.2f}'
            }),
            hide_index=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="world_cup_2026_results.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No simulation results available. Run a simulation first!")
        if st.button("Go to Simulation"):
            st.experimental_rerun()


def show_training():
    """Model training page."""
    st.header("🔧 Model Training")
    
    st.markdown("""
    This page allows you to retrain the prediction models with updated data.
    """)
    
    st.markdown("---")
    
    # ELO calculation
    st.subheader("1. Generate ELO Ratings")
    st.write("Calculate ELO ratings and form metrics from historical match data.")
    
    if st.button("🔄 Generate ELO Ratings"):
        with st.spinner("Calculating ELO ratings..."):
            try:
                result = subprocess.run(
                    ["python", "src/elo/generate_elo.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    st.success("✅ ELO ratings generated successfully!")
                    st.code(result.stdout)
                else:
                    st.error("❌ Error generating ELO ratings")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Train ML models
    st.subheader("2. Train ML Models (XGBoost & CatBoost)")
    st.write("Train gradient boosting models for goal prediction.")
    
    if st.button("🤖 Train XGBoost & CatBoost"):
        with st.spinner("Training models (this may take a few minutes)..."):
            try:
                result = subprocess.run(
                    ["python", "src/train/train_models.py"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    st.success("✅ Models trained successfully!")
                    st.code(result.stdout)
                else:
                    st.error("❌ Error training models")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Train Poisson & Dixon-Coles
    st.subheader("3. Train Statistical Models (Poisson & Dixon-Coles)")
    st.write("Train Poisson regression and estimate Dixon-Coles parameters.")
    
    if st.button("📊 Train Poisson & Dixon-Coles"):
        with st.spinner("Training statistical models..."):
            try:
                result = subprocess.run(
                    ["python", "src/prediction/train_all.py"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    st.success("✅ Statistical models trained successfully!")
                    st.code(result.stdout)
                else:
                    st.error("❌ Error training statistical models")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    st.info("💡 Tip: Train models in order (1 → 2 → 3) for best results.")


def show_about():
    """About page."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## FIFA World Cup 2026 Simulator
    
    A comprehensive Monte Carlo simulator for the 2026 FIFA World Cup using advanced statistical models and machine learning.
    
    ### Features
    
    - **48-team tournament simulation** with official 2026 groups
    - **Multiple prediction models**: XGBoost, CatBoost, Poisson, Dixon-Coles
    - **Monte Carlo simulation** for probability estimation
    - **Detailed match-by-match reporting**
    - **Interactive visualizations**
    
    ### Methodology
    
    1. **ELO Rating System**
       - Custom implementation with competition-weighted K-factors
       - Home advantage adjustment
       - Form calculation
    
    2. **Machine Learning Models**
       - XGBoost and CatBoost regressors
       - Trained on 25,000+ historical matches
       - Poisson loss function for count data
    
    3. **Statistical Models**
       - Poisson regression (GLM baseline)
       - Dixon-Coles adjustment for low-scoring games
    
    4. **Tournament Simulation**
       - Group stage with third-place qualification logic
       - Knockout rounds with extra time and penalties
       - Monte Carlo aggregation of results
    
    ### Data Sources
    
    - Historical international match results (2000-2026)
    - Official 2026 World Cup groups
    - Team ELO ratings and form metrics
    
    ### Technology Stack
    
    - **Python** - Core language
    - **Streamlit** - Web interface
    - **Pandas/NumPy** - Data processing
    - **Scikit-learn** - Poisson regression
    - **XGBoost/CatBoost** - Gradient boosting
    - **Plotly** - Interactive visualizations
    
    ### Author
    
    Built as a demonstration of sports analytics and predictive modeling.
    
    ### Version
    
    1.0.0 - June 2026
    """)


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    Path("./results").mkdir(exist_ok=True)
    
    main()
