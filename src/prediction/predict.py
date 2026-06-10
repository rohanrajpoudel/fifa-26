"""
Unified Prediction Interface
Combines Poisson Regression, XGBoost, CatBoost, and Dixon-Coles models
for comprehensive goal and match outcome predictions.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class GoalPredictor:
    """
    Unified interface for all goal prediction models.
    Supports: Poisson Regression, XGBoost, CatBoost, and Dixon-Coles adjustments.
    """
    
    def __init__(self, models_dir='./models'):
        """
        Initialize predictor and load all available models.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available prediction models."""
        print("Loading prediction models...")
        
        # Load XGBoost models
        try:
            self.models['xgb_home'] = joblib.load(self.models_dir / 'xgb_home.pkl')
            self.models['xgb_away'] = joblib.load(self.models_dir / 'xgb_away.pkl')
            self.models['xgb_features'] = joblib.load(self.models_dir / 'feature_columns.pkl')
            print("  ✓ XGBoost models loaded")
        except FileNotFoundError:
            print("  ✗ XGBoost models not found")
        
        # Load CatBoost models
        try:
            self.models['cat_home'] = joblib.load(self.models_dir / 'cat_home.pkl')
            self.models['cat_away'] = joblib.load(self.models_dir / 'cat_away.pkl')
            self.models['cat_features'] = joblib.load(self.models_dir / 'cat_features.pkl')
            print("  ✓ CatBoost models loaded")
        except FileNotFoundError:
            print("  ✗ CatBoost models not found")
        
        # Load Poisson model
        try:
            from .poisson_model import PoissonGoalPredictor
            self.models['poisson'] = PoissonGoalPredictor.load(self.models_dir / 'poisson_model.pkl')
            print("  ✓ Poisson model loaded")
        except FileNotFoundError:
            print("  ✗ Poisson model not found")
        
        # Load Dixon-Coles parameters
        try:
            from .dixon_coles import DixonColesModel
            self.models['dc_xgb'] = DixonColesModel.load(self.models_dir / 'dixon_coles_xgb.pkl')
            print("  ✓ Dixon-Coles (XGBoost) loaded")
        except FileNotFoundError:
            print("  ✗ Dixon-Coles (XGBoost) not found")
        
        try:
            from .dixon_coles import DixonColesModel
            self.models['dc_cat'] = DixonColesModel.load(self.models_dir / 'dixon_coles_cat.pkl')
            print("  ✓ Dixon-Coles (CatBoost) loaded")
        except FileNotFoundError:
            print("  ✗ Dixon-Coles (CatBoost) not found")
    
    def prepare_xgb_features(self, df):
        """Prepare features for XGBoost model."""
        features = [
            'home_elo_before', 'away_elo_before',
            'home_form_before', 'away_form_before',
            'elo_diff', 'form_diff',
            'home_attack_5', 'away_attack_5',
            'home_defense_5', 'away_defense_5',
            'home_attack_10', 'away_attack_10',
            'home_defense_10', 'away_defense_10',
            'home_win_rate_5', 'away_win_rate_5',
            'home_win_rate_10', 'away_win_rate_10',
            'home_goal_diff_5', 'away_goal_diff_5',
            'home_goal_diff_10', 'away_goal_diff_10',
            'neutral_site'
        ]
        
        cat_features = ['competition', 'home_confederation', 'away_confederation']
        
        X = df[features + cat_features].copy()
        X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
        X_encoded = X_encoded.reindex(columns=self.models['xgb_features'], fill_value=0)
        
        return X_encoded
    
    def prepare_cat_features(self, df):
        """Prepare features for CatBoost model."""
        features = [
            'home_elo_before', 'away_elo_before',
            'home_form_before', 'away_form_before',
            'elo_diff', 'form_diff',
            'home_attack_5', 'away_attack_5',
            'home_defense_5', 'away_defense_5',
            'home_attack_10', 'away_attack_10',
            'home_defense_10', 'away_defense_10',
            'home_win_rate_5', 'away_win_rate_5',
            'home_win_rate_10', 'away_win_rate_10',
            'home_goal_diff_5', 'away_goal_diff_5',
            'home_goal_diff_10', 'away_goal_diff_10',
            'neutral_site', 'competition',
            'home_confederation', 'away_confederation'
        ]
        
        return df[features].copy()
    
    def predict_xgboost(self, df):
        """Predict goals using XGBoost models."""
        if 'xgb_home' not in self.models:
            raise ValueError("XGBoost models not loaded")
        
        X = self.prepare_xgb_features(df)
        home_goals = np.clip(self.models['xgb_home'].predict(X), 0.01, None)
        away_goals = np.clip(self.models['xgb_away'].predict(X), 0.01, None)
        
        return home_goals, away_goals
    
    def predict_catboost(self, df):
        """Predict goals using CatBoost models."""
        if 'cat_home' not in self.models:
            raise ValueError("CatBoost models not loaded")
        
        X = self.prepare_cat_features(df)
        home_goals = np.clip(self.models['cat_home'].predict(X), 0.01, None)
        away_goals = np.clip(self.models['cat_away'].predict(X), 0.01, None)
        
        return home_goals, away_goals
    
    def predict_poisson(self, df):
        """Predict goals using Poisson regression."""
        if 'poisson' not in self.models:
            raise ValueError("Poisson model not loaded")
        
        return self.models['poisson'].predict(df)
    
    def predict_all(self, df):
        """
        Get predictions from all available models.
        
        Args:
            df: DataFrame with match features
            
        Returns:
            DataFrame with predictions from all models
        """
        results = pd.DataFrame()
        results['match_id'] = df.get('match_id', range(len(df)))
        
        # Add match information if available
        if 'home_team' in df.columns:
            results['home_team'] = df['home_team'].values
        if 'away_team' in df.columns:
            results['away_team'] = df['away_team'].values
        if 'date' in df.columns:
            results['date'] = df['date'].values
        
        # XGBoost predictions
        if 'xgb_home' in self.models:
            home, away = self.predict_xgboost(df)
            results['xgb_home'] = home
            results['xgb_away'] = away
        
        # CatBoost predictions
        if 'cat_home' in self.models:
            home, away = self.predict_catboost(df)
            results['cat_home'] = home
            results['cat_away'] = away
        
        # Poisson predictions
        if 'poisson' in self.models:
            home, away = self.predict_poisson(df)
            results['poisson_home'] = home
            results['poisson_away'] = away
        
        # Ensemble average
        home_cols = [c for c in results.columns if c.endswith('_home')]
        away_cols = [c for c in results.columns if c.endswith('_away')]
        
        if home_cols:
            results['ensemble_home'] = results[home_cols].mean(axis=1)
        if away_cols:
            results['ensemble_away'] = results[away_cols].mean(axis=1)
        
        return results
    
    def predict_with_dixon_coles(self, df, base_model='xgb'):
        """
        Predict using Dixon-Coles adjustment.
        
        Args:
            df: DataFrame with match features
            base_model: Base model to use ('xgb', 'cat', 'poisson')
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Get base predictions
        if base_model == 'xgb':
            home_lambda, away_lambda = self.predict_xgboost(df)
            dc_model = self.models.get('dc_xgb')
        elif base_model == 'cat':
            home_lambda, away_lambda = self.predict_catboost(df)
            dc_model = self.models.get('dc_cat')
        elif base_model == 'poisson':
            home_lambda, away_lambda = self.predict_poisson(df)
            dc_model = self.models.get('dc_xgb')  # Use XGBoost DC params as default
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        if dc_model is None:
            raise ValueError(f"Dixon-Coles model for {base_model} not loaded")
        
        # Get Dixon-Coles adjusted predictions
        adj_home, adj_away = dc_model.predict_expected_goals(home_lambda, away_lambda)
        
        # Get outcome probabilities
        outcomes = dc_model.predict_match_outcomes(home_lambda, away_lambda)
        
        # Get scoreline probabilities
        scoreline_probs = dc_model.predict_probabilities(home_lambda, away_lambda)
        
        results = {
            'base_home': home_lambda,
            'base_away': away_lambda,
            'dc_home': adj_home,
            'dc_away': adj_away,
            'home_win_prob': outcomes['home_win_prob'].values,
            'draw_prob': outcomes['draw_prob'].values,
            'away_win_prob': outcomes['away_win_prob'].values,
            'scoreline_probs': scoreline_probs,
            'rho': dc_model.rho
        }
        
        return results
    
    def predict_match(self, home_team, away_team, features, use_dixon_coles=True):
        """
        Comprehensive prediction for a single match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            features: Dictionary or DataFrame with match features
            use_dixon_coles: Whether to apply Dixon-Coles adjustment
            
        Returns:
            Dictionary with comprehensive predictions
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()
        
        # Get all base predictions
        all_preds = self.predict_all(df)
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': {}
        }
        
        # Add base model predictions
        if 'xgb_home' in all_preds.columns:
            result['predictions']['xgboost'] = {
                'home_goals': all_preds['xgb_home'].iloc[0],
                'away_goals': all_preds['xgb_away'].iloc[0]
            }
        
        if 'cat_home' in all_preds.columns:
            result['predictions']['catboost'] = {
                'home_goals': all_preds['cat_home'].iloc[0],
                'away_goals': all_preds['cat_away'].iloc[0]
            }
        
        if 'poisson_home' in all_preds.columns:
            result['predictions']['poisson'] = {
                'home_goals': all_preds['poisson_home'].iloc[0],
                'away_goals': all_preds['poisson_away'].iloc[0]
            }
        
        if 'ensemble_home' in all_preds.columns:
            result['predictions']['ensemble'] = {
                'home_goals': all_preds['ensemble_home'].iloc[0],
                'away_goals': all_preds['ensemble_away'].iloc[0]
            }
        
        # Add Dixon-Coles predictions
        if use_dixon_coles:
            if 'dc_xgb' in self.models:
                dc_xgb = self.predict_with_dixon_coles(df, base_model='xgb')
                result['predictions']['dixon_coles_xgb'] = {
                    'home_goals': dc_xgb['dc_home'][0],
                    'away_goals': dc_xgb['dc_away'][0],
                    'home_win_prob': dc_xgb['home_win_prob'][0],
                    'draw_prob': dc_xgb['draw_prob'][0],
                    'away_win_prob': dc_xgb['away_win_prob'][0],
                    'rho': dc_xgb['rho']
                }
            
            if 'dc_cat' in self.models:
                dc_cat = self.predict_with_dixon_coles(df, base_model='cat')
                result['predictions']['dixon_coles_cat'] = {
                    'home_goals': dc_cat['dc_home'][0],
                    'away_goals': dc_cat['dc_away'][0],
                    'home_win_prob': dc_cat['home_win_prob'][0],
                    'draw_prob': dc_cat['draw_prob'][0],
                    'away_win_prob': dc_cat['away_win_prob'][0],
                    'rho': dc_cat['rho']
                }
        
        return result
    
    def get_most_likely_scorelines(self, scoreline_probs, n=5):
        """
        Get the n most likely scorelines from probability matrix.
        
        Args:
            scoreline_probs: Probability matrix [max_goals+1, max_goals+1]
            n: Number of top scorelines to return
            
        Returns:
            List of (home_goals, away_goals, probability) tuples
        """
        # Flatten and get top N
        flat_indices = np.argsort(scoreline_probs.ravel())[::-1][:n]
        scorelines = []
        
        for idx in flat_indices:
            home = idx // scoreline_probs.shape[1]
            away = idx % scoreline_probs.shape[1]
            prob = scoreline_probs[home, away]
            scorelines.append((home, away, prob))
        
        return scorelines


def example_prediction():
    """Example usage of the unified predictor."""
    print("="*70)
    print("FIFA Match Prediction - Unified Interface")
    print("="*70)
    
    # Initialize predictor
    predictor = GoalPredictor()
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv('./data/matches_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Add derived features
    if 'elo_diff' not in df.columns:
        df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    if 'form_diff' not in df.columns:
        df['form_diff'] = df['home_form_before'] - df['away_form_before']
    
    # Get recent matches
    test = df[df['date'] >= '2024-01-01'].head(5)
    
    print("\n" + "="*70)
    print("Sample Predictions")
    print("="*70)
    
    for idx, row in test.iterrows():
        print(f"\n{row['home_team']} vs {row['away_team']}")
        print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
        print(f"Competition: {row['competition']}")
        print(f"Actual Score: {int(row['home_goals'])} - {int(row['away_goals'])}")
        
        # Get predictions
        pred = predictor.predict_match(
            home_team=row['home_team'],
            away_team=row['away_team'],
            features=row.to_frame().T
        )
        
        print("\nPredicted Goals:")
        for model_name, preds in pred['predictions'].items():
            if 'dixon_coles' in model_name:
                print(f"  {model_name:20s}: {preds['home_goals']:.2f} - {preds['away_goals']:.2f} "
                      f"(W: {preds['home_win_prob']:.2%}, D: {preds['draw_prob']:.2%}, "
                      f"L: {preds['away_win_prob']:.2%})")
            else:
                print(f"  {model_name:20s}: {preds['home_goals']:.2f} - {preds['away_goals']:.2f}")
        
        print("-" * 70)


if __name__ == "__main__":
    example_prediction()
