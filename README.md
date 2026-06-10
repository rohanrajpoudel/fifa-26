# FIFA World Cup 2026 Prediction System

Tournament prediction using ELO ratings, Gradient Boosting, Poisson goal models, and Monte Carlo simulation.

## Project Overview

A statistical modeling system for predicting FIFA World Cup 2026 outcomes by combining historical match data, custom ELO rating system, machine learning models, and probabilistic simulation.

## Architecture

```
Historical Match Data (2000-2026)
        ↓
Data Cleaning & Feature Engineering
        ↓
Custom ELO Rating System
        ↓
Machine Learning Models (XGBoost/LightGBM)
        ↓
Poisson/Dixon-Coles Goal Distribution
        ↓
Monte Carlo Tournament Simulator
        ↓
Win Probabilities & Visualizations
```

## Current Implementation Status

### ✅ Completed

**Data Pipeline**
- Historical match data collection (2000-2026)
- Data cleaning and preprocessing
- Match result extraction
- Team statistics aggregation

**ELO Rating System**
- Custom ELO implementation with competition-based K-factors:
  - World Cup: K=60
  - Continental Cups: K=40
  - Nations League: K=35
  - Qualifiers: K=30
  - Friendlies: K=20
- Home advantage adjustment (+30 ELO)
- Goal difference margin multiplier
- Form calculation (last 10 matches, weighted)
- Rating history tracking

**Generated Datasets**
- `data/cleaned_matches.csv` - cleaned historical matches (2000-2026)
- `data/teams_with_elo.csv` - all teams with current ELO and form
- `data/matches_with_features.csv` - matches with pre-match features
- `data/elo_history.csv` - complete ELO progression

**Machine Learning Models**
- XGBoost regressors for home/away goals (Poisson loss)
- CatBoost regressors for home/away goals (Poisson loss)
- Poisson regression (statistical GLM baseline)
- Dixon-Coles adjustments for low-scoring games

**Goal Prediction System**
- Unified prediction interface (`src/prediction/`)
- Multiple model ensemble
- Scoreline probability distributions
- Match outcome probabilities
- Comprehensive evaluation framework

### 🚧 In Progress / To Do

**Phase 1: Baseline Models** ✓ COMPLETE
- [x] Train match outcome classifier (Win/Draw/Loss)
- [x] Train XGBoost/CatBoost models with Poisson loss
- [x] Model evaluation and calibration
- [x] Feature importance analysis

**Phase 2: Goal Prediction** ✓ COMPLETE
- [x] Implement Poisson regression for expected goals
- [x] Dixon-Coles model for low-scoring adjustments
- [x] Goal distribution validation
- [x] Expected goals (xG) prediction per team
- [x] Unified prediction interface
- [x] Comprehensive model evaluation

**Phase 3: Tournament Simulation**
- [ ] Group stage simulator (12 groups of 4 teams)
- [ ] Knockout stage bracket generator
- [ ] Round of 32 implementation
- [ ] Monte Carlo simulation engine (10,000+ runs)
- [ ] Tournament outcome probabilities

**Phase 4: Model Explainability**
- [ ] SHAP value analysis
- [ ] Feature importance visualization
- [ ] Calibration plots
- [ ] Prediction confidence intervals

**Phase 5: Validation & Backtesting**
- [ ] 2022 World Cup retrospective prediction
- [ ] Historical tournament accuracy
- [ ] Model performance metrics
- [ ] Comparison with bookmaker odds

**Phase 6: Visualization & Dashboard**
- [ ] Interactive Streamlit dashboard
- [ ] Team comparison tool
- [ ] Match predictor interface
- [ ] Tournament probability charts
- [ ] ELO progression plots

## Project Structure

```
fifa-26/
├── data/                           # Generated datasets
│   ├── cleaned_matches.csv         # ✓ Cleaned historical matches
│   ├── teams_with_elo.csv          # ✓ All teams with ELO ratings
│   ├── matches_with_features.csv   # ✓ Matches with pre-match features
│   └── elo_history.csv             # ✓ Complete ELO progression
├── rawData/                        # Source data (gitignored)
│   ├── results.csv                 # Historical match results
│   ├── goalscorers.csv             # Goal scorer details
│   ├── shootouts.csv               # Penalty shootout data
│   └── former_names.csv            # Team name changes
├── src/
│   ├── data_processing/            # Data cleaning pipeline
│   │   └── clean_data.py
│   ├── elo/                        # ELO rating system
│   │   └── generate_elo.py
│   ├── train/                      # Model training
│   │   └── train_models.py         # XGBoost/CatBoost training
│   └── prediction/                 # ✓ Goal prediction (Phase 3)
│       ├── poisson_model.py        # Poisson regression
│       ├── dixon_coles.py          # Dixon-Coles model
│       ├── predict.py              # Unified prediction interface
│       ├── evaluate_models.py      # Model evaluation
│       └── train_all.py            # Training pipeline
├── models/                         # Trained models
│   ├── xgb_home.pkl, xgb_away.pkl
│   ├── cat_home.pkl, cat_away.pkl
│   ├── poisson_model.pkl           # ✓ New
│   ├── dixon_coles_xgb.pkl         # ✓ New
│   └── dixon_coles_cat.pkl         # ✓ New
├── examples/                       # Usage examples
│   └── predict_match.py
├── gptResponse/                    # Project planning documentation
│   ├── 00.1.md                     # Architecture recommendations
│   └── 00.2.md                     # Project roadmap
├── PHASE3_COMPLETE.md              # ✓ Implementation details
├── QUICK_START_PHASE3.md           # ✓ Quick reference guide
├── IMPLEMENTATION_SUMMARY.md       # ✓ Summary document
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Clean Historical Data

```bash
python src/data_processing/clean_data.py
```

Processes raw match data and generates cleaned datasets filtered for 2000-2026.

### 2. Generate ELO Ratings

```bash
python src/elo/generate_elo.py
```

Calculates ELO ratings, form metrics, and generates feature-rich datasets.

### 3. Train Models

```bash
# Train XGBoost/CatBoost models
python src/train/train_models.py

# Train Poisson and Dixon-Coles models
python src/prediction/train_all.py
```

### 4. Make Predictions

```python
from src.prediction.predict import GoalPredictor

predictor = GoalPredictor()
result = predictor.predict_match(
    home_team='Brazil',
    away_team='Argentina',
    features=match_features
)
```


## World Cup 2026 Teams (48 Teams)

Algeria, Argentina, Australia, Austria, Belgium, Bosnia and Herzegovina, Brazil, Canada, Cape Verde, Colombia, Croatia, Curacao, Czech Republic, DR Congo, Ecuador, Egypt, England, France, Germany, Ghana, Haiti, Iran, Iraq, Ivory Coast, Japan, Jordan, Mexico, Morocco, Netherlands, New Zealand, Norway, Panama, Paraguay, Portugal, Qatar, Saudi Arabia, Scotland, Senegal, South Africa, South Korea, Spain, Sweden, Switzerland, Tunisia, Turkey, United States, Uruguay, Uzbekistan

## Methodology

### ELO Rating System

Custom implementation with competition weighting:
- Base rating: 1500
- K-factor varies by competition importance
- Home advantage: +30 ELO (neutral venue = 0)
- Goal margin multiplier: 1.0-2.0x based on score difference
- Form: Weighted average of last 10 match results

### Expected Modeling Pipeline

1. **Feature Engineering**: ELO, form, goals scored/conceded, competition type
2. **Classification**: Win/Draw/Loss probability using gradient boosting
3. **Regression**: Expected goals per team (Poisson distribution)
4. **Simulation**: Monte Carlo tournament runs
5. **Aggregation**: Winner probability, qualification odds, expected position

## Future Enhancements

- Squad market value integration
- Player availability tracking
- Travel distance impact
- Confederation strength factors
- Weather/venue conditions
- Real-time data updates
- API for predictions
- Docker deployment

## Data Sources

- Historical international match results
- Goal scorers database
- Penalty shootout records
- Team name changes

## Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Statistics**: scipy (Poisson distributions)
- **Visualization**: matplotlib, seaborn
- **Notebook**: Jupyter

## License

Educational project for Data Science portfolio development.

## References

- World Football ELO Ratings
- FIFA World Cup 2026 Format
- Dixon-Coles Model for Football Prediction
- Poisson Distribution for Goal Modeling
