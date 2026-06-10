"""
Dixon-Coles Model for Goal Prediction
Implements the Dixon-Coles adjustment to Poisson model for low-scoring matches.
Accounts for correlation between home and away goals in scorelines like 0-0, 1-0, 0-1, 1-1.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies 
in the Football Betting Market"
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
import joblib
import warnings

warnings.filterwarnings('ignore')


class DixonColesModel:
    """
    Dixon-Coles model for football match prediction.
    Extends Poisson regression with correlation parameter (rho) for low-scoring games.
    """
    
    def __init__(self, base_model=None):
        """
        Initialize Dixon-Coles model.
        
        Args:
            base_model: Base model for predicting goal rates (lambda_home, lambda_away)
                       Can be Poisson, XGBoost, or CatBoost model
        """
        self.base_model = base_model
        self.rho = 0.0  # Correlation parameter
        self.home_advantage = 0.0  # Additional home advantage parameter
        
    def fit_rho(self, home_lambda, away_lambda, home_goals, away_goals):
        """
        Estimate the rho correlation parameter using maximum likelihood.
        
        Args:
            home_lambda: Predicted home goal rates
            away_lambda: Predicted away goal rates
            home_goals: Actual home goals scored
            away_goals: Actual away goals scored
            
        Returns:
            Estimated rho value
        """
        print("Estimating Dixon-Coles rho parameter...")
        
        def negative_log_likelihood(rho):
            """Negative log-likelihood for optimization."""
            total_ll = 0.0
            
            for i in range(len(home_goals)):
                h_goals = int(home_goals[i])
                a_goals = int(away_goals[i])
                h_lambda = home_lambda[i]
                a_lambda = away_lambda[i]
                
                # Base Poisson probability
                prob = poisson.pmf(h_goals, h_lambda) * poisson.pmf(a_goals, a_lambda)
                
                # Apply Dixon-Coles adjustment for low-scoring games
                tau = self._tau_correction(h_goals, a_goals, h_lambda, a_lambda, rho)
                prob = prob * tau
                
                # Add to log-likelihood (with numerical stability)
                total_ll += np.log(max(prob, 1e-10))
            
            return -total_ll
        
        # Optimize rho parameter
        result = minimize(
            negative_log_likelihood,
            x0=[0.0],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)]
        )
        
        self.rho = result.x[0]
        print(f"  Estimated rho: {self.rho:.6f}")
        
        return self.rho
    
    @staticmethod
    def _tau_correction(home_goals, away_goals, home_lambda, away_lambda, rho):
        """
        Dixon-Coles correlation adjustment factor.
        Modifies probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines.
        
        Args:
            home_goals: Home team goals
            away_goals: Away team goals
            home_lambda: Expected home goals
            away_lambda: Expected away goals
            rho: Correlation parameter
            
        Returns:
            Correction factor tau
        """
        # No adjustment for scorelines with 2+ goals by either team
        if home_goals > 1 or away_goals > 1:
            return 1.0
        
        # Adjustment for low-scoring games
        if home_goals == 0 and away_goals == 0:
            # 0-0: typically occurs more often than independent Poisson predicts
            tau = 1 - home_lambda * away_lambda * rho
        elif home_goals == 0 and away_goals == 1:
            # 0-1: adjust for correlation
            tau = 1 + home_lambda * rho
        elif home_goals == 1 and away_goals == 0:
            # 1-0: adjust for correlation
            tau = 1 + away_lambda * rho
        elif home_goals == 1 and away_goals == 1:
            # 1-1: adjust for correlation
            tau = 1 - rho
        else:
            tau = 1.0
        
        return tau
    
    def predict_probabilities(self, home_lambda, away_lambda, max_goals=10):
        """
        Predict scoreline probabilities using Dixon-Coles adjusted Poisson.
        
        Args:
            home_lambda: Expected home goals (array)
            away_lambda: Expected away goals (array)
            max_goals: Maximum goals to consider
            
        Returns:
            3D array of probabilities [n_matches, max_goals+1, max_goals+1]
        """
        n_matches = len(home_lambda)
        prob_matrix = np.zeros((n_matches, max_goals + 1, max_goals + 1))
        
        for i in range(n_matches):
            h_lambda = home_lambda[i]
            a_lambda = away_lambda[i]
            
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    # Base Poisson probability
                    prob = poisson.pmf(h, h_lambda) * poisson.pmf(a, a_lambda)
                    
                    # Apply Dixon-Coles correction
                    tau = self._tau_correction(h, a, h_lambda, a_lambda, self.rho)
                    prob_matrix[i, h, a] = prob * tau
            
            # Normalize probabilities to sum to 1
            prob_matrix[i] = prob_matrix[i] / prob_matrix[i].sum()
        
        return prob_matrix
    
    def predict_match_outcomes(self, home_lambda, away_lambda):
        """
        Predict match outcome probabilities (home win, draw, away win).
        
        Args:
            home_lambda: Expected home goals
            away_lambda: Expected away goals
            
        Returns:
            DataFrame with outcome probabilities
        """
        prob_matrix = self.predict_probabilities(home_lambda, away_lambda)
        n_matches = prob_matrix.shape[0]
        
        outcomes = np.zeros((n_matches, 3))
        
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
    
    def predict_expected_goals(self, home_lambda, away_lambda):
        """
        Calculate expected goals from probability distribution.
        
        Args:
            home_lambda: Base expected home goals
            away_lambda: Base expected away goals
            
        Returns:
            Tuple of (adjusted_home_goals, adjusted_away_goals)
        """
        prob_matrix = self.predict_probabilities(home_lambda, away_lambda)
        n_matches = prob_matrix.shape[0]
        max_goals = prob_matrix.shape[1] - 1
        
        expected_home = np.zeros(n_matches)
        expected_away = np.zeros(n_matches)
        
        for i in range(n_matches):
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    expected_home[i] += h * prob_matrix[i, h, a]
                    expected_away[i] += a * prob_matrix[i, h, a]
        
        return expected_home, expected_away
    
    def evaluate(self, home_lambda, away_lambda, actual_home, actual_away):
        """
        Evaluate Dixon-Coles predictions.
        
        Args:
            home_lambda: Predicted home goal rates
            away_lambda: Predicted away goal rates
            actual_home: Actual home goals
            actual_away: Actual away goals
            
        Returns:
            Dictionary of metrics
        """
        # Get adjusted expected goals
        adj_home, adj_away = self.predict_expected_goals(home_lambda, away_lambda)
        
        metrics = {
            'home': {
                'mae': mean_absolute_error(actual_home, adj_home),
                'rmse': np.sqrt(mean_squared_error(actual_home, adj_home)),
                'mean_pred': np.mean(adj_home),
                'mean_actual': np.mean(actual_home)
            },
            'away': {
                'mae': mean_absolute_error(actual_away, adj_away),
                'rmse': np.sqrt(mean_squared_error(actual_away, adj_away)),
                'mean_pred': np.mean(adj_away),
                'mean_actual': np.mean(actual_away)
            },
            'rho': self.rho
        }
        
        return metrics
    
    def compare_scoreline_frequencies(self, home_lambda, away_lambda, actual_home, actual_away):
        """
        Compare actual vs predicted frequencies for key scorelines.
        Shows the improvement from Dixon-Coles adjustment.
        
        Args:
            home_lambda: Predicted home goal rates
            away_lambda: Predicted away goal rates
            actual_home: Actual home goals
            actual_away: Actual away goals
            
        Returns:
            DataFrame comparing actual and predicted scoreline frequencies
        """
        # Get Dixon-Coles adjusted probabilities
        prob_matrix_dc = self.predict_probabilities(home_lambda, away_lambda, max_goals=5)
        
        # Calculate independent Poisson probabilities
        n_matches = len(home_lambda)
        prob_matrix_ind = np.zeros((n_matches, 6, 6))
        for i in range(n_matches):
            for h in range(6):
                for a in range(6):
                    prob_matrix_ind[i, h, a] = (
                        poisson.pmf(h, home_lambda[i]) * poisson.pmf(a, away_lambda[i])
                    )
        
        # Key scorelines to compare
        scorelines = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
        
        results = []
        for h, a in scorelines:
            # Actual frequency
            actual_freq = np.mean((actual_home == h) & (actual_away == a))
            
            # Independent Poisson prediction
            ind_poisson_freq = np.mean(prob_matrix_ind[:, h, a])
            
            # Dixon-Coles prediction
            dc_freq = np.mean(prob_matrix_dc[:, h, a])
            
            results.append({
                'scoreline': f"{h}-{a}",
                'actual': actual_freq,
                'independent_poisson': ind_poisson_freq,
                'dixon_coles': dc_freq,
                'poisson_error': abs(actual_freq - ind_poisson_freq),
                'dc_error': abs(actual_freq - dc_freq)
            })
        
        return pd.DataFrame(results)
    
    def save(self, filepath):
        """Save Dixon-Coles parameters."""
        model_data = {
            'rho': self.rho,
            'home_advantage': self.home_advantage
        }
        joblib.dump(model_data, filepath)
        print(f"Dixon-Coles parameters saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load Dixon-Coles parameters."""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.rho = model_data['rho']
        instance.home_advantage = model_data.get('home_advantage', 0.0)
        print(f"Dixon-Coles parameters loaded from {filepath}")
        return instance


def train_dixon_coles_with_xgboost():
    """
    Train Dixon-Coles model using XGBoost predictions as base rates.
    """
    print("="*70)
    print("Training Dixon-Coles Model with XGBoost Base")
    print("="*70)
    
    # Load XGBoost models
    print("\nLoading XGBoost models...")
    xgb_home = joblib.load('./models/xgb_home.pkl')
    xgb_away = joblib.load('./models/xgb_away.pkl')
    feature_columns = joblib.load('./models/feature_columns.pkl')
    
    # Load validation predictions for rho estimation
    print("Loading validation predictions...")
    val_preds = pd.read_csv('./models/results/validation_predictions.csv')
    
    # Initialize Dixon-Coles model
    dc_model = DixonColesModel()
    
    # Fit rho parameter using XGBoost predictions
    dc_model.fit_rho(
        home_lambda=val_preds['pred_home_xgb'].values,
        away_lambda=val_preds['pred_away_xgb'].values,
        home_goals=val_preds['actual_home'].values,
        away_goals=val_preds['actual_away'].values
    )
    
    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    metrics = dc_model.evaluate(
        home_lambda=val_preds['pred_home_xgb'].values,
        away_lambda=val_preds['pred_away_xgb'].values,
        actual_home=val_preds['actual_home'].values,
        actual_away=val_preds['actual_away'].values
    )
    
    print(f"\nHome Goals (XGBoost + Dixon-Coles):")
    print(f"  MAE: {metrics['home']['mae']:.4f}")
    print(f"  RMSE: {metrics['home']['rmse']:.4f}")
    print(f"  Mean Predicted: {metrics['home']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['home']['mean_actual']:.4f}")
    
    print(f"\nAway Goals (XGBoost + Dixon-Coles):")
    print(f"  MAE: {metrics['away']['mae']:.4f}")
    print(f"  RMSE: {metrics['away']['rmse']:.4f}")
    print(f"  Mean Predicted: {metrics['away']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['away']['mean_actual']:.4f}")
    
    # Compare scoreline frequencies
    print("\n" + "="*70)
    print("Scoreline Frequency Comparison")
    print("="*70)
    scoreline_comp = dc_model.compare_scoreline_frequencies(
        home_lambda=val_preds['pred_home_xgb'].values,
        away_lambda=val_preds['pred_away_xgb'].values,
        actual_home=val_preds['actual_home'].values,
        actual_away=val_preds['actual_away'].values
    )
    print(scoreline_comp.to_string(index=False))
    
    # Save model
    dc_model.save('./models/dixon_coles_xgb.pkl')
    
    return dc_model


def train_dixon_coles_with_catboost():
    """
    Train Dixon-Coles model using CatBoost predictions as base rates.
    """
    print("\n" + "="*70)
    print("Training Dixon-Coles Model with CatBoost Base")
    print("="*70)
    
    # Load validation predictions
    val_preds = pd.read_csv('./models/results/validation_predictions.csv')
    
    # Initialize Dixon-Coles model
    dc_model = DixonColesModel()
    
    # Fit rho parameter using CatBoost predictions
    dc_model.fit_rho(
        home_lambda=val_preds['pred_home_cat'].values,
        away_lambda=val_preds['pred_away_cat'].values,
        home_goals=val_preds['actual_home'].values,
        away_goals=val_preds['actual_away'].values
    )
    
    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    metrics = dc_model.evaluate(
        home_lambda=val_preds['pred_home_cat'].values,
        away_lambda=val_preds['pred_away_cat'].values,
        actual_home=val_preds['actual_home'].values,
        actual_away=val_preds['actual_away'].values
    )
    
    print(f"\nHome Goals (CatBoost + Dixon-Coles):")
    print(f"  MAE: {metrics['home']['mae']:.4f}")
    print(f"  RMSE: {metrics['home']['rmse']:.4f}")
    print(f"  Mean Predicted: {metrics['home']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['home']['mean_actual']:.4f}")
    
    print(f"\nAway Goals (CatBoost + Dixon-Coles):")
    print(f"  MAE: {metrics['away']['mae']:.4f}")
    print(f"  RMSE: {metrics['away']['rmse']:.4f}")
    print(f"  Mean Predicted: {metrics['away']['mean_pred']:.4f}")
    print(f"  Mean Actual: {metrics['away']['mean_actual']:.4f}")
    
    # Compare scoreline frequencies
    print("\n" + "="*70)
    print("Scoreline Frequency Comparison")
    print("="*70)
    scoreline_comp = dc_model.compare_scoreline_frequencies(
        home_lambda=val_preds['pred_home_cat'].values,
        away_lambda=val_preds['pred_away_cat'].values,
        actual_home=val_preds['actual_home'].values,
        actual_away=val_preds['actual_away'].values
    )
    print(scoreline_comp.to_string(index=False))
    
    # Save model
    dc_model.save('./models/dixon_coles_cat.pkl')
    
    return dc_model


if __name__ == "__main__":
    # Train Dixon-Coles with both XGBoost and CatBoost
    dc_xgb = train_dixon_coles_with_xgboost()
    dc_cat = train_dixon_coles_with_catboost()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"XGBoost Dixon-Coles rho: {dc_xgb.rho:.6f}")
    print(f"CatBoost Dixon-Coles rho: {dc_cat.rho:.6f}")
