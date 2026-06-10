"""
Comprehensive Model Evaluation and Comparison
Evaluates Poisson, XGBoost, CatBoost, and Dixon-Coles models on test data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation of all goal prediction models.
    Compares accuracy, calibration, and scoreline prediction quality.
    """
    
    def __init__(self, models_dir='./models', data_path='./data/matches_with_features.csv'):
        """
        Initialize evaluator.
        
        Args:
            models_dir: Directory containing saved models
            data_path: Path to matches data
        """
        self.models_dir = Path(models_dir)
        self.data_path = data_path
        self.test_data = None
        self.results = {}
    
    def load_test_data(self):
        """Load and prepare test data."""
        print("Loading test data...")
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
        
        # Add derived features
        if 'elo_diff' not in df.columns:
            df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
        if 'form_diff' not in df.columns:
            df['form_diff'] = df['home_form_before'] - df['away_form_before']
        
        # Use 2023+ as test set
        self.test_data = df[df['date'] >= '2023-01-01'].copy()
        
        print(f"Test set: {len(self.test_data)} matches")
        print(f"Date range: {self.test_data['date'].min()} to {self.test_data['date'].max()}")
        
        return self.test_data
    
    def evaluate_xgboost(self):
        """Evaluate XGBoost models."""
        print("\n" + "="*70)
        print("Evaluating XGBoost Models")
        print("="*70)
        
        try:
            # Load models
            xgb_home = joblib.load(self.models_dir / 'xgb_home.pkl')
            xgb_away = joblib.load(self.models_dir / 'xgb_away.pkl')
            feature_columns = joblib.load(self.models_dir / 'feature_columns.pkl')
            
            # Prepare features
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
            
            X = self.test_data[features + cat_features].copy()
            X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
            X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)
            
            # Predict
            pred_home = np.clip(xgb_home.predict(X_encoded), 0.01, None)
            pred_away = np.clip(xgb_away.predict(X_encoded), 0.01, None)
            
            # Evaluate
            actual_home = self.test_data['home_goals'].values
            actual_away = self.test_data['away_goals'].values
            
            metrics = self._calculate_metrics(pred_home, pred_away, actual_home, actual_away)
            self.results['xgboost'] = {
                'pred_home': pred_home,
                'pred_away': pred_away,
                'metrics': metrics
            }
            
            self._print_metrics("XGBoost", metrics)
            
        except Exception as e:
            print(f"Error evaluating XGBoost: {e}")
    
    def evaluate_catboost(self):
        """Evaluate CatBoost models."""
        print("\n" + "="*70)
        print("Evaluating CatBoost Models")
        print("="*70)
        
        try:
            # Load models
            cat_home = joblib.load(self.models_dir / 'cat_home.pkl')
            cat_away = joblib.load(self.models_dir / 'cat_away.pkl')
            
            # Prepare features
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
            
            X = self.test_data[features].copy()
            
            # Predict
            pred_home = np.clip(cat_home.predict(X), 0.01, None)
            pred_away = np.clip(cat_away.predict(X), 0.01, None)
            
            # Evaluate
            actual_home = self.test_data['home_goals'].values
            actual_away = self.test_data['away_goals'].values
            
            metrics = self._calculate_metrics(pred_home, pred_away, actual_home, actual_away)
            self.results['catboost'] = {
                'pred_home': pred_home,
                'pred_away': pred_away,
                'metrics': metrics
            }
            
            self._print_metrics("CatBoost", metrics)
            
        except Exception as e:
            print(f"Error evaluating CatBoost: {e}")
    
    def evaluate_poisson(self):
        """Evaluate Poisson regression model."""
        print("\n" + "="*70)
        print("Evaluating Poisson Regression")
        print("="*70)
        
        try:
            from .poisson_model import PoissonGoalPredictor
            
            # Load model
            model = PoissonGoalPredictor.load(self.models_dir / 'poisson_model.pkl')
            
            # Predict
            pred_home, pred_away = model.predict(self.test_data)
            
            # Evaluate
            actual_home = self.test_data['home_goals'].values
            actual_away = self.test_data['away_goals'].values
            
            metrics = self._calculate_metrics(pred_home, pred_away, actual_home, actual_away)
            self.results['poisson'] = {
                'pred_home': pred_home,
                'pred_away': pred_away,
                'metrics': metrics
            }
            
            self._print_metrics("Poisson", metrics)
            
        except Exception as e:
            print(f"Error evaluating Poisson: {e}")
    
    def evaluate_dixon_coles(self):
        """Evaluate Dixon-Coles adjustments."""
        print("\n" + "="*70)
        print("Evaluating Dixon-Coles Models")
        print("="*70)
        
        try:
            from .dixon_coles import DixonColesModel
            
            # XGBoost + Dixon-Coles
            if 'xgboost' in self.results:
                dc_xgb = DixonColesModel.load(self.models_dir / 'dixon_coles_xgb.pkl')
                
                adj_home, adj_away = dc_xgb.predict_expected_goals(
                    self.results['xgboost']['pred_home'],
                    self.results['xgboost']['pred_away']
                )
                
                actual_home = self.test_data['home_goals'].values
                actual_away = self.test_data['away_goals'].values
                
                metrics = self._calculate_metrics(adj_home, adj_away, actual_home, actual_away)
                metrics['rho'] = dc_xgb.rho
                
                self.results['dixon_coles_xgb'] = {
                    'pred_home': adj_home,
                    'pred_away': adj_away,
                    'metrics': metrics
                }
                
                self._print_metrics("Dixon-Coles (XGBoost)", metrics)
            
            # CatBoost + Dixon-Coles
            if 'catboost' in self.results:
                dc_cat = DixonColesModel.load(self.models_dir / 'dixon_coles_cat.pkl')
                
                adj_home, adj_away = dc_cat.predict_expected_goals(
                    self.results['catboost']['pred_home'],
                    self.results['catboost']['pred_away']
                )
                
                actual_home = self.test_data['home_goals'].values
                actual_away = self.test_data['away_goals'].values
                
                metrics = self._calculate_metrics(adj_home, adj_away, actual_home, actual_away)
                metrics['rho'] = dc_cat.rho
                
                self.results['dixon_coles_cat'] = {
                    'pred_home': adj_home,
                    'pred_away': adj_away,
                    'metrics': metrics
                }
                
                self._print_metrics("Dixon-Coles (CatBoost)", metrics)
                
        except Exception as e:
            print(f"Error evaluating Dixon-Coles: {e}")
    
    def evaluate_ensemble(self):
        """Evaluate ensemble predictions."""
        print("\n" + "="*70)
        print("Evaluating Ensemble")
        print("="*70)
        
        try:
            # Average all base models (excluding Dixon-Coles)
            home_preds = []
            away_preds = []
            
            for model_name in ['xgboost', 'catboost', 'poisson']:
                if model_name in self.results:
                    home_preds.append(self.results[model_name]['pred_home'])
                    away_preds.append(self.results[model_name]['pred_away'])
            
            if home_preds:
                pred_home = np.mean(home_preds, axis=0)
                pred_away = np.mean(away_preds, axis=0)
                
                actual_home = self.test_data['home_goals'].values
                actual_away = self.test_data['away_goals'].values
                
                metrics = self._calculate_metrics(pred_home, pred_away, actual_home, actual_away)
                self.results['ensemble'] = {
                    'pred_home': pred_home,
                    'pred_away': pred_away,
                    'metrics': metrics
                }
                
                self._print_metrics("Ensemble", metrics)
                
        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
    
    def _calculate_metrics(self, pred_home, pred_away, actual_home, actual_away):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'home_mae': mean_absolute_error(actual_home, pred_home),
            'away_mae': mean_absolute_error(actual_away, pred_away),
            'home_rmse': np.sqrt(mean_squared_error(actual_home, pred_home)),
            'away_rmse': np.sqrt(mean_squared_error(actual_away, pred_away)),
            'home_poisson_dev': mean_poisson_deviance(actual_home, pred_home),
            'away_poisson_dev': mean_poisson_deviance(actual_away, pred_away),
            'mean_pred_home': np.mean(pred_home),
            'mean_pred_away': np.mean(pred_away),
            'mean_actual_home': np.mean(actual_home),
            'mean_actual_away': np.mean(actual_away)
        }
        
        # Calculate combined metrics
        metrics['combined_mae'] = (metrics['home_mae'] + metrics['away_mae']) / 2
        metrics['combined_rmse'] = (metrics['home_rmse'] + metrics['away_rmse']) / 2
        
        # Outcome prediction accuracy
        pred_result = np.sign(pred_home - pred_away)
        actual_result = np.sign(actual_home - actual_away)
        metrics['outcome_accuracy'] = np.mean(pred_result == actual_result)
        
        return metrics
    
    def _print_metrics(self, model_name, metrics):
        """Print metrics in a formatted way."""
        print(f"\n{model_name} Results:")
        print(f"  Home Goals:")
        print(f"    MAE: {metrics['home_mae']:.4f}")
        print(f"    RMSE: {metrics['home_rmse']:.4f}")
        print(f"    Poisson Deviance: {metrics['home_poisson_dev']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_pred_home']:.4f}")
        print(f"    Mean Actual: {metrics['mean_actual_home']:.4f}")
        
        print(f"  Away Goals:")
        print(f"    MAE: {metrics['away_mae']:.4f}")
        print(f"    RMSE: {metrics['away_rmse']:.4f}")
        print(f"    Poisson Deviance: {metrics['away_poisson_dev']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_pred_away']:.4f}")
        print(f"    Mean Actual: {metrics['mean_actual_away']:.4f}")
        
        print(f"  Combined:")
        print(f"    Average MAE: {metrics['combined_mae']:.4f}")
        print(f"    Average RMSE: {metrics['combined_rmse']:.4f}")
        print(f"    Outcome Accuracy: {metrics['outcome_accuracy']:.2%}")
        
        if 'rho' in metrics:
            print(f"  Dixon-Coles rho: {metrics['rho']:.6f}")
    
    def compare_models(self):
        """Generate comparison table of all models."""
        print("\n" + "="*70)
        print("Model Comparison Summary")
        print("="*70)
        
        comparison_data = []
        
        for model_name, model_data in self.results.items():
            metrics = model_data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Home MAE': metrics['home_mae'],
                'Away MAE': metrics['away_mae'],
                'Avg MAE': metrics['combined_mae'],
                'Avg RMSE': metrics['combined_rmse'],
                'Outcome Acc': metrics['outcome_accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Avg MAE')
        
        print("\n" + comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save to CSV
        output_path = self.models_dir / 'results' / 'model_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\nComparison saved to {output_path}")
        
        return comparison_df
    
    def analyze_scoreline_accuracy(self):
        """Analyze prediction accuracy for common scorelines."""
        print("\n" + "="*70)
        print("Scoreline Prediction Analysis")
        print("="*70)
        
        actual_home = self.test_data['home_goals'].values
        actual_away = self.test_data['away_goals'].values
        
        # Common scorelines
        scorelines = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
        
        print("\nActual Scoreline Frequencies:")
        for h, a in scorelines:
            freq = np.mean((actual_home == h) & (actual_away == a))
            count = np.sum((actual_home == h) & (actual_away == a))
            print(f"  {h}-{a}: {freq:.3%} ({count} matches)")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Load data
        self.load_test_data()
        
        # Evaluate all models
        self.evaluate_xgboost()
        self.evaluate_catboost()
        self.evaluate_poisson()
        self.evaluate_dixon_coles()
        self.evaluate_ensemble()
        
        # Compare models
        self.compare_models()
        
        # Analyze scorelines
        self.analyze_scoreline_accuracy()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        return self.results


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation()
