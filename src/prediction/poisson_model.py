"""
Poisson Regression Model for Goal Prediction
Implements traditional statistical Poisson GLM for expected goals prediction.
Complementary to the XGBoost/CatBoost models trained with Poisson loss.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
import joblib
import warnings

warnings.filterwarnings('ignore')


class PoissonGoalPredictor:
    """
    Statistical Poisson regression model for goal prediction.
    Uses GLM with log link function - the classical approach for count data.
    """
    
    def __init__(self, alpha=1.0, max_iter=300):
        """
        Initialize Poisson regression models for home and away goals.
        
        Args:
            alpha: Regularization strength (L2 penalty)
            max_iter: Maximum iterations for optimization
        """
        self.home_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.away_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Extract and prepare features for Poisson regression.
        
        Args:
            df: DataFrame with match data and features
            
        Returns:
            Prepared feature matrix
        """
        # Core features that are always available
        base_features = [
            'home_elo_before', 'away_elo_before',
            'home_form_before', 'away_form_before',
            'elo_diff', 'form_diff',
            'neutral_site'
        ]
        
        # Optional rolling features (if available from training data)
        rolling_features = [
            'home_attack_5', 'away_attack_5',
            'home_defense_5', 'away_defense_5',
            'home_attack_10', 'away_attack_10',
            'home_defense_10', 'away_defense_10',
            'home_win_rate_5', 'away_win_rate_5',
            'home_win_rate_10', 'away_win_rate_10',
            'home_goal_diff_5', 'away_goal_diff_5',
            'home_goal_diff_10', 'away_goal_diff_10'
        ]
        
        # Check which rolling features are available
        available_rolling = [f for f in rolling_features if f in df.columns]
        features = base_features + available_rolling
        
        # Handle categorical features if available
        cat_features = []
        if 'competition' in df.columns:
            cat_features.append('competition')
        if 'home_confederation' in df.columns:
            cat_features.append('home_confederation')
        if 'away_confederation' in df.columns:
            cat_features.append('away_confederation')
        
        X = df[features + cat_features].copy()
        
        if cat_features:
            X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
        else:
            X_encoded = X
        
        return X_encoded
    
    def fit(self, train_df, val_df=None):
        """
        Train Poisson regression models on historical match data.
        
        Args:
            train_df: Training data with features and goals
            val_df: Optional validation data for monitoring
        """
        print("Training Poisson Regression Models...")
        
        # Prepare features
        X_train = self.prepare_features(train_df)
        self.feature_names = X_train.columns.tolist()
        
        # Scale features for better convergence
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        y_train_home = train_df['home_goals'].values
        y_train_away = train_df['away_goals'].values
        
        # Train home goals model
        print("  Training home goals model...")
        self.home_model.fit(X_train_scaled, y_train_home)
        
        # Train away goals model
        print("  Training away goals model...")
        self.away_model.fit(X_train_scaled, y_train_away)
        
        # Validation metrics if provided
        if val_df is not None:
            self._evaluate_on_validation(val_df)
    
    def _evaluate_on_validation(self, val_df):
        """Internal method to evaluate on validation set."""
        X_val = self.prepare_features(val_df)
        X_val = X_val.reindex(columns=self.feature_names, fill_value=0)
        X_val_scaled = self.scaler.transform(X_val)
        
        y_val_home = val_df['home_goals'].values
        y_val_away = val_df['away_goals'].values
        
        pred_home = self.home_model.predict(X_val_scaled)
        pred_away = self.away_model.predict(X_val_scaled)
        
        print("\n  Validation Metrics:")
        print(f"    Home - MAE: {mean_absolute_error(y_val_home, pred_home):.4f}, "
              f"RMSE: {np.sqrt(mean_squared_error(y_val_home, pred_home)):.4f}, "
              f"Poisson Dev: {mean_poisson_deviance(y_val_home, pred_home):.4f}")
        print(f"    Away - MAE: {mean_absolute_error(y_val_away, pred_away):.4f}, "
              f"RMSE: {np.sqrt(mean_squared_error(y_val_away, pred_away)):.4f}, "
              f"Poisson Dev: {mean_poisson_deviance(y_val_away, pred_away):.4f}")
    
    def predict(self, df):
        """
        Predict expected goals for home and away teams.
        
        Args:
            df: DataFrame with match features
            
        Returns:
            Tuple of (home_goals, away_goals) predictions
        """
        X = self.prepare_features(df)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X)
        
        home_goals = self.home_model.predict(X_scaled)
        away_goals = self.away_model.predict(X_scaled)
        
        return home_goals, away_goals
    
    def predict_probabilities(self, df, max_goals=10):
        """
        Predict probability distribution over possible scorelines.
        Uses independent Poisson distributions for each team.
        
        Args:
            df: DataFrame with match features
            max_goals: Maximum goals to consider (default 10)
            
        Returns:
            3D array of shape (n_matches, max_goals+1, max_goals+1)
            where arr[i, h, a] = P(home=h, away=a) for match i
        """
        home_lambda, away_lambda = self.predict(df)
        n_matches = len(df)
        
        # Initialize probability matrix
        prob_matrix = np.zeros((n_matches, max_goals + 1, max_goals + 1))
        
        # Calculate independent Poisson probabilities
        for i in range(n_matches):
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    # P(H=h) * P(A=a) under independence assumption
                    prob_home = self._poisson_prob(home_goals, home_lambda[i])
                    prob_away = self._poisson_prob(away_goals, away_lambda[i])
                    prob_matrix[i, home_goals, away_goals] = prob_home * prob_away
        
        return prob_matrix
    
    @staticmethod
    def _poisson_prob(k, lambda_):
        """Calculate Poisson probability P(X=k) for rate lambda."""
        from scipy.stats import poisson
        return poisson.pmf(k, lambda_)
    
    def predict_match_outcome(self, df):
        """
        Predict match outcome probabilities (win/draw/loss).
        
        Args:
            df: DataFrame with match features
            
        Returns:
            DataFrame with columns: home_win_prob, draw_prob, away_win_prob
        """
        prob_matrix = self.predict_probabilities(df)
        n_matches = prob_matrix.shape[0]
        
        outcomes = np.zeros((n_matches, 3))  # home_win, draw, away_win
        
        for i in range(n_matches):
            # Sum probabilities for each outcome
            home_win = np.sum(np.tril(prob_matrix[i], -1))  # home > away
            draw = np.sum(np.diag(prob_matrix[i]))          # home == away
            away_win = np.sum(np.triu(prob_matrix[i], 1))   # home < away
            
            outcomes[i] = [home_win, draw, away_win]
        
        return pd.DataFrame(
            outcomes,
            columns=['home_win_prob', 'draw_prob', 'away_win_prob']
        )
    
    def evaluate(self, test_df):
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_df: Test data with features and actual goals
            
        Returns:
            Dictionary of evaluation metrics
        """
        home_pred, away_pred = self.predict(test_df)
        
        y_test_home = test_df['home_goals'].values
        y_test_away = test_df['away_goals'].values
        
        metrics = {
            'home': {
                'mae': mean_absolute_error(y_test_home, home_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_home, home_pred)),
                'poisson_deviance': mean_poisson_deviance(y_test_home, home_pred),
                'mean_pred': np.mean(home_pred),
                'mean_actual': np.mean(y_test_home)
            },
            'away': {
                'mae': mean_absolute_error(y_test_away, away_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_away, away_pred)),
                'poisson_deviance': mean_poisson_deviance(y_test_away, away_pred),
                'mean_pred': np.mean(away_pred),
                'mean_actual': np.mean(y_test_away)
            }
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to disk."""
        model_data = {
            'home_model': self.home_model,
            'away_model': self.away_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.home_model = model_data['home_model']
        instance.away_model = model_data['away_model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
        return instance


def train_poisson_model(data_path='./data/matches_with_features.csv'):
    """
    Complete training pipeline for Poisson regression model.
    
    Args:
        data_path: Path to matches data with features
        
    Returns:
        Trained PoissonGoalPredictor instance
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    
    # Add derived features if not present
    if 'elo_diff' not in df.columns:
        df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    if 'form_diff' not in df.columns:
        df['form_diff'] = df['home_form_before'] - df['away_form_before']
    
    # Time-based split
    train = df[df['date'] < '2019-01-01']
    val = df[(df['date'] >= '2019-01-01') & (df['date'] < '2023-01-01')]
    test = df[df['date'] >= '2023-01-01']
    
    print(f"\nData splits:")
    print(f"  Train: {len(train)} matches")
    print(f"  Val: {len(val)} matches")
    print(f"  Test: {len(test)} matches")
    
    # Train model
    model = PoissonGoalPredictor(alpha=0.1)
    model.fit(train, val)
    
    # Test evaluation
    print("\nTest Set Evaluation:")
    metrics = model.evaluate(test)
    
    print(f"\nHome Goals:")
    print(f"  MAE: {metrics['home']['mae']:.4f}")
    print(f"  RMSE: {metrics['home']['rmse']:.4f}")
    print(f"  Poisson Deviance: {metrics['home']['poisson_deviance']:.4f}")
    print(f"  Mean Predicted: {metrics['home']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['home']['mean_actual']:.4f}")
    
    print(f"\nAway Goals:")
    print(f"  MAE: {metrics['away']['mae']:.4f}")
    print(f"  RMSE: {metrics['away']['rmse']:.4f}")
    print(f"  Poisson Deviance: {metrics['away']['poisson_deviance']:.4f}")
    print(f"  Mean Predicted: {metrics['away']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['away']['mean_actual']:.4f}")
    
    return model


if __name__ == "__main__":
    # Train and save Poisson model
    model = train_poisson_model()
    model.save('./models/poisson_model.pkl')
