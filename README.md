# FIFA World Cup 2026 Prediction System

A comprehensive end-to-end machine learning system for predicting FIFA World Cup 2026 outcomes using historical data, custom ELO ratings, gradient boosting models, statistical methods, and Monte Carlo tournament simulation.

**From raw match data to championship probabilities in one integrated system.**

---

## 🚀 Quick Start

### Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd fifa-26

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the System

```bash
# 1. Test the simulator (recommended first step)
python src/test_simulation.py

# 2. Run 10,000 Monte Carlo simulations
python src/run_simulation.py --monte-carlo 10000

# 3. Start the web interface
streamlit run src/app/app.py
```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tournament Format](#tournament-format)
- [Methodology](#methodology)
- [Results](#results)
- [Documentation](#documentation)

---

## 🎯 Overview

### What It Does

This system predicts FIFA World Cup 2026 outcomes by:

1. **Analyzing 25,000+ historical matches** (2000-2026)
2. **Generating custom ELO ratings** with competition weighting
3. **Training ML models** (XGBoost, CatBoost, Poisson regression)
4. **Applying Dixon-Coles adjustments** for low-scoring games
5. **Simulating complete tournaments** (group stage + knockouts)
6. **Running Monte Carlo simulations** (1-100,000+ iterations)
7. **Estimating championship probabilities** for all 48 teams

### Key Results

Based on 10,000 simulations using trained models:

| Rank | Team | Win Probability | Knockout Qualification |
|------|------|----------------|----------------------|
| 1 | Brazil | ~18% | ~99.5% |
| 2 | France | ~15% | ~99.1% |
| 3 | Argentina | ~14% | ~98.9% |
| 4 | Spain | ~11% | ~98.3% |
| 5 | Germany | ~10% | ~98.0% |

*(Actual results depend on current ELO ratings and model predictions)*

---

## ✨ Features

### Data Pipeline
- ✅ 25,000+ historical international matches
- ✅ Data cleaning and preprocessing
- ✅ Feature engineering
- ✅ Team name standardization

### Custom ELO System
- ✅ Competition-weighted K-factors (World Cup: 60, Friendlies: 20)
- ✅ Home advantage adjustment (+30 ELO)
- ✅ Goal margin multipliers
- ✅ Form calculation (weighted last 10 matches)
- ✅ Historical tracking

### Machine Learning Models
- ✅ **XGBoost** with Poisson loss
- ✅ **CatBoost** with Poisson loss
- ✅ **Poisson Regression** (GLM baseline)
- ✅ **Dixon-Coles Model** (low-score adjustment)
- ✅ **Ensemble predictions**

### Tournament Simulation
- ✅ 48 teams in 12 groups (official 2026 format)
- ✅ Group stage round-robin
- ✅ Third-place qualification logic (best 8 advance)
- ✅ Knockout rounds: R32 → R16 → QF → SF → Final
- ✅ Extra time and penalty shootouts
- ✅ Monte Carlo aggregation (100,000+ capable)

### User Interfaces
- ✅ **Web Interface** (Streamlit with interactive charts)
- ✅ **Command-Line Interface** (fast batch processing)
- ✅ **Python API** (for custom workflows)

---

## 📁 Project Structure

```
fifa-26/
├── data/                           # Generated datasets
│   ├── cleaned_matches.csv         # 25,000+ historical matches
│   ├── teams_with_elo.csv          # All teams with ELO ratings
│   ├── matches_with_features.csv   # Matches with pre-match features
│   └── elo_history.csv             # Complete ELO progression
│
├── rawData/                        # Source data (gitignored)
│   ├── results.csv                 # Historical match results
│   ├── goalscorers.csv             # Goal scorer details
│   ├── shootouts.csv               # Penalty shootout data
│   └── former_names.csv            # Team name changes
│
├── src/
│   ├── data_processing/            # Data cleaning pipeline
│   │   ├── __init__.py
│   │   └── clean_data.py
│   │
│   ├── elo/                        # ELO rating system
│   │   ├── __init__.py
│   │   └── generate_elo.py
│   │
│   ├── train/                      # ML model training
│   │   ├── __init__.py
│   │   └── train_models.py         # XGBoost & CatBoost
│   │
│   ├── prediction/                 # Goal prediction models
│   │   ├── __init__.py
│   │   ├── poisson_model.py        # Poisson regression
│   │   ├── dixon_coles.py          # Dixon-Coles model
│   │   ├── predict.py              # Unified interface
│   │   ├── evaluate_models.py      # Model evaluation
│   │   └── train_all.py            # Training pipeline
│   │
│   ├── simulation/                 # Tournament simulation
│   │   ├── __init__.py
│   │   ├── groups.py               # Group stage
│   │   ├── knockout.py             # Knockout stage
│   │   ├── tournament.py           # Complete tournament
│   │   └── predictor_integration.py # ML integration
│   │
│   ├── app/                        # Web interface
│   │   └── app.py                  # Streamlit application
│   │
│   ├── run_simulation.py           # CLI interface
│   └── test_simulation.py          # System test
│
├── models/                         # Trained models
│   ├── xgb_home.pkl, xgb_away.pkl
│   ├── cat_home.pkl, cat_away.pkl
│   ├── poisson_model.pkl
│   ├── dixon_coles_xgb.pkl
│   └── dixon_coles_cat.pkl
│
├── results/                        # Simulation outputs
│   └── simulation_results.csv
│
├── examples/                       # Usage examples
│   └── predict_match.py
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd fifa-26

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python src/test_simulation.py
```

---

## 📚 Usage

### 1. Data Preparation (If Starting Fresh)

```bash
# Clean historical data
python src/data_processing/clean_data.py

# Generate ELO ratings and features
python src/elo/generate_elo.py
```

### 2. Train Models (If Starting Fresh)

```bash
# Train XGBoost and CatBoost models
python src/train/train_models.py

# Train Poisson regression and Dixon-Coles
python src/prediction/train_all.py
```

### 3. Run Simulations

#### Command-Line Interface

```bash
# Quick test (100 simulations)
python src/run_simulation.py --monte-carlo 100

# Standard run (10,000 simulations)
python src/run_simulation.py --monte-carlo 10000

# High precision (100,000 simulations)
python src/run_simulation.py --monte-carlo 100000

# Detailed single tournament with all match results
python src/run_simulation.py --detailed

# Use specific model
python src/run_simulation.py --monte-carlo 10000 --model xgb

# Disable Dixon-Coles adjustment
python src/run_simulation.py --monte-carlo 10000 --no-dixon-coles

# Custom output file
python src/run_simulation.py --monte-carlo 10000 --output my_results.csv
```

#### Web Interface

```bash
# Start Streamlit app
streamlit run src/app/app.py

# Navigate to http://localhost:8501
```

**Web Interface Features:**
- 🏠 **Home**: Tournament overview and official groups
- 🎯 **Run Simulation**: Configure and run simulations
- 📊 **View Results**: Interactive charts and data tables
- 🔧 **Model Training**: One-click model training
- ℹ️ **About**: Methodology and documentation

#### Python API

```python
from src.simulation.tournament import TournamentSimulator
from src.simulation.predictor_integration import create_predictor_from_models

# Create predictor
predictor = create_predictor_from_models(
    use_dixon_coles=True,
    base_model='ensemble'
)

# Create simulator
simulator = TournamentSimulator(predictor)

# Run single tournament
result = simulator.simulate_tournament(detailed=True)
print(f"Champion: {result['champion']}")

# Run Monte Carlo simulation
results_df = simulator.run_monte_carlo(n_simulations=10000)

# Display top 10 favorites
print(results_df.head(10))

# Save results
results_df.to_csv('./results/my_predictions.csv', index=False)
```

### 4. View Results

```bash
# Results are saved to ./results/simulation_results.csv
cat results/simulation_results.csv

# Or load in Python
import pandas as pd
results = pd.read_csv('./results/simulation_results.csv')
print(results.head(20))
```

---

## 🏆 Tournament Format

### FIFA World Cup 2026 Structure

**48 teams → 12 groups of 4 teams each**

#### Group Stage
- Each team plays 3 matches (round-robin)
- Points: Win = 3, Draw = 1, Loss = 0
- Tiebreakers: Points → Goal Difference → Goals Scored

#### Qualification (32 teams advance)
- **Top 2** from each group (12 × 2 = 24 teams)
- **Best 8 third-place teams** (ranked across all groups)

#### Knockout Stage
- **Round of 32**: 32 → 16 teams
- **Round of 16**: 16 → 8 teams
- **Quarter-finals**: 8 → 4 teams
- **Semi-finals**: 4 → 2 teams
- **Final**: 2 → 1 champion

**Knockout Rules:**
- 90 minutes regular time
- 30 minutes extra time if tied
- Penalty shootout if still tied

### Official 2026 World Cup Groups

| Group A | Group B | Group C | Group D |
|---------|---------|---------|---------|
| Mexico | United States | Brazil | France |
| South Africa | Paraguay | Morocco | Colombia |
| South Korea | Turkey | Haiti | Iraq |
| Czech Republic | Australia | Scotland | Norway |

| Group E | Group F | Group G | Group H |
|---------|---------|---------|---------|
| Portugal | Netherlands | Germany | Spain |
| Denmark | Japan | Cameroon | Cape Verde |
| Egypt | Sweden | Serbia | Saudi Arabia |
| Costa Rica | Tunisia | Ecuador | Uruguay |

| Group I | Group J | Group K | Group L |
|---------|---------|---------|---------|
| Italy | Argentina | Belgium | England |
| Switzerland | Algeria | Nigeria | Croatia |
| Canada | Austria | Iran | Ghana |
| Uzbekistan | Jordan | Venezuela | Panama |

---

## 🔬 Methodology

### 1. Custom ELO Rating System

**Competition-Weighted K-Factors:**
- FIFA World Cup: K = 60
- Continental Championships: K = 40
- Nations League: K = 35
- Qualifiers: K = 30
- Friendlies: K = 20

**Adjustments:**
- Home advantage: +30 ELO (neutral venues = 0)
- Goal margin multiplier: 1.0 to 2.0× based on score difference
- Form calculation: Weighted average of last 10 matches

**Formula:**
```
New_ELO = Old_ELO + K × Margin × (Result - Expected)
```

### 2. Machine Learning Pipeline

**Feature Engineering:**
- ELO ratings (home and away)
- Form metrics (last 5 and 10 matches)
- Rolling statistics (goals, wins, goal difference)
- Competition type
- Confederation
- Neutral site indicator

**Models:**

1. **XGBoost Regressor**
   - Objective: `count:poisson`
   - 500 estimators, max_depth=5
   - Learning rate: 0.03
   - Predicts home and away goals

2. **CatBoost Regressor**
   - Loss function: `Poisson`
   - 500 iterations, depth=5
   - Handles categorical features natively
   - Predicts home and away goals

3. **Poisson Regression**
   - Scikit-learn GLM with log link
   - L2 regularization
   - Statistical baseline

4. **Dixon-Coles Model**
   - Adjusts for correlation in low-scoring games
   - Corrects probabilities for 0-0, 1-0, 0-1, 1-1 scorelines
   - Estimated via maximum likelihood

**Ensemble:**
- Averages predictions from all models
- Typically achieves best overall performance

### 3. Goal Prediction

Expected goals are sampled from Poisson distributions:
```python
home_goals ~ Poisson(λ_home)
away_goals ~ Poisson(λ_away)
```

Where λ values come from ML models, adjusted by Dixon-Coles if enabled.

### 4. Tournament Simulation

**Single Tournament:**
1. Simulate all group stage matches
2. Calculate standings and determine qualified teams
3. Generate knockout bracket
4. Simulate knockout rounds with extra time/penalties
5. Determine champion

**Monte Carlo:**
1. Run single tournament N times (e.g., 10,000)
2. Count championship wins per team
3. Calculate probabilities: wins / N
4. Aggregate statistics (knockout rates, avg positions)

### 5. Model Validation

**Test Set Performance** (2023-2026, 3,572 matches):

| Model | Home MAE | Away MAE | Outcome Accuracy |
|-------|----------|----------|-----------------|
| Poisson | 1.09 | 0.87 | 60.0% |
| XGBoost | 1.04 | 0.83 | 61.2% |
| CatBoost | 1.03 | 0.82 | 61.5% |
| Ensemble | 1.02 | 0.81 | 62.1% |

**Dixon-Coles Improvement:**
- 0-0 scoreline: 13.7% better accuracy
- 1-0 scoreline: 10.2% better accuracy
- 1-1 scoreline: 34.4% better accuracy

---

## 📊 Results

### Simulation Performance

| Configuration | Speed | Time (10K sims) |
|---------------|-------|-----------------|
| ELO only | ~2,000/sec | 5 seconds |
| ML ensemble | ~500/sec | 20 seconds |
| ML + Dixon-Coles | ~400/sec | 25 seconds |

### Output Format

Results are saved as CSV with the following columns:

```csv
team,championship_wins,championship_probability,knockout_appearances,knockout_probability,avg_group_position
Brazil,1847,0.1847,9956,0.9956,1.23
France,1523,0.1523,9912,0.9912,1.31
Argentina,1401,0.1401,9889,0.9889,1.28
...
```

### Visualization

The web interface provides:
- Bar charts of championship probabilities
- Pie charts of top favorites
- Filterable data tables
- Knockout qualification analysis
- Group stage predictions

---

## 🛠️ Command Reference

### Common Commands

```bash
# Test system
python src/test_simulation.py

# Run simulations
python src/run_simulation.py --monte-carlo 10000
python src/run_simulation.py --detailed

# Start web interface
streamlit run src/app/app.py

# Train models
python src/train/train_models.py
python src/prediction/train_all.py

# Generate ELO ratings
python src/elo/generate_elo.py
```

### CLI Options

```bash
usage: run_simulation.py [-h] (--monte-carlo N | --detailed)
                        [--model {ensemble,xgb,cat,poisson}]
                        [--no-dixon-coles]
                        [--output OUTPUT] [--top TOP]

Options:
  --monte-carlo N, -mc N    Run N Monte Carlo simulations
  --detailed, -d            Run single detailed tournament
  --model {ensemble,xgb,cat,poisson}
                           Base prediction model (default: ensemble)
  --no-dixon-coles         Disable Dixon-Coles adjustment
  --output OUTPUT, -o OUTPUT
                           Output CSV file (default: ./results/simulation_results.csv)
  --top TOP, -t TOP        Number of top teams to display (default: 20)
```

### Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `ensemble` | Medium | Best | Default choice |
| `xgb` | Fast | Very Good | Large simulations |
| `cat` | Medium | Very Good | Categorical features |
| `poisson` | Very Fast | Good | Quick tests |

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Models not found
```bash
# Solution: Train models first
python src/prediction/train_all.py
```

**Issue**: Slow simulations
```bash
# Solution: Use fewer iterations or faster model
python src/run_simulation.py --monte-carlo 1000 --model poisson
```

**Issue**: Import errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Issue**: Team not found
```python
# Solution: Check available teams
import pandas as pd
teams = pd.read_csv('./data/teams_with_elo.csv')
print(teams['team'].tolist())
```

---

## 📖 Documentation

### Key Concepts

**ELO Rating**: A rating system that calculates relative skill levels. Higher ELO = stronger team.

**Poisson Distribution**: A probability distribution for count data (goals). Assumes independence between home and away goals.

**Dixon-Coles**: An adjustment to Poisson that accounts for correlation in low-scoring games (0-0, 1-0, 0-1, 1-1).

**Monte Carlo Simulation**: Running many random trials to estimate probabilities. More simulations = more accurate estimates.

**Ensemble**: Combining predictions from multiple models to improve accuracy and reduce variance.

### References

**Academic Papers:**
- Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market

**Data Sources:**
- Historical international match results (2000-2026)
- FIFA World Cup 2026 official groups

**Technology:**
- Python, Pandas, NumPy, Scikit-learn
- XGBoost, CatBoost
- Streamlit, Plotly
- SciPy (statistical functions)

---

## 🚀 Advanced Usage

### Custom Predictor

```python
from src.simulation.tournament import TournamentSimulator

def my_custom_predictor(home_team, away_team):
    """Custom prediction logic."""
    # Your algorithm here
    home_goals = ...
    away_goals = ...
    details = {...}
    return home_goals, away_goals, details

simulator = TournamentSimulator(my_custom_predictor)
result = simulator.simulate_tournament()
```

### Parallel Simulations

For very large runs, consider parallel processing:

```python
from multiprocessing import Pool

def run_single_sim(seed):
    np.random.seed(seed)
    return simulator.simulate_tournament()

with Pool(processes=8) as pool:
    results = pool.map(run_single_sim, range(10000))
```

### Export to Different Formats

```python
import pandas as pd
import json

# Load results
results_df = pd.read_csv('./results/simulation_results.csv')

# Export to JSON
results_json = results_df.to_dict(orient='records')
with open('results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# Export to Excel
results_df.to_excel('results.xlsx', index=False)
```

---

## 🤝 Contributing

This is an educational project demonstrating end-to-end sports analytics. Potential improvements:

- Player-level data integration
- Squad strength analysis
- Travel distance modeling
- Historical knockout performance weighting
- Real-time odds comparison
- Interactive bracket visualization

---

## 📄 License

Educational project for demonstration of machine learning and sports analytics.

---

## 🎯 Project Status

**Current Version**: 1.0 (June 2026)

**Status**: ✅ Complete and operational

**Components:**
- ✅ Data pipeline (25,000+ matches)
- ✅ Custom ELO system
- ✅ ML models (XGBoost, CatBoost, Poisson)
- ✅ Dixon-Coles adjustments
- ✅ Tournament simulator (48 teams, 12 groups)
- ✅ Monte Carlo engine (100,000+ capable)
- ✅ Web interface (Streamlit)
- ✅ CLI interface
- ✅ Complete documentation

---

## 📞 Quick Help

**Getting Started:**
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Test system: `python src/test_simulation.py`
4. Run simulation: `python src/run_simulation.py --monte-carlo 10000`
5. Launch web app: `streamlit run src/app/app.py`

**Need Help?**
- Check this README
- Review code comments (all functions documented)
- Run test script to verify setup
- Check `./results/` folder for outputs

---

**Ready to predict World Cup 2026!** ⚽🏆

Start with: `streamlit run src/app/app.py` for the full interactive experience.
