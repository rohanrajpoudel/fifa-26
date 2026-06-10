"""
Integration between prediction models and tournament simulator.
Connects trained ML models with simulation engine.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TournamentPredictor:
    """
    Wrapper around prediction models for tournament simulation.
    Handles feature preparation and model selection.
    """
    
    def __init__(self, use_dixon_coles: bool = True, base_model: str = 'ensemble'):
        """
        Initialize tournament predictor.
        
        Args:
            use_dixon_coles: Whether to use Dixon-Coles adjustments
            base_model: Base model to use ('xgb', 'cat', 'poisson', 'ensemble')
        """
        self.use_dixon_coles = use_dixon_coles
        self.base_model = base_model
        
        # Load team data
        self.teams_df = pd.read_csv('./data/teams_with_elo.csv')
        self.team_stats = self._prepare_team_stats()
        
        # Load prediction models
        self.predictor = self._load_predictor()
    
    def _prepare_team_stats(self) -> Dict[str, Dict]:
        """Prepare team statistics dictionary."""
        team_stats = {}
        
        for _, row in self.teams_df.iterrows():
            # Handle NaN/empty confederation values
            conf = row.get('confederation', 'Unknown')
            if pd.isna(conf) or conf == '':
                conf = 'Unknown'
            
            team_stats[row['team']] = {
                'elo': row['current_elo'],
                'form': row['current_form'],
                'confederation': str(conf)  # Ensure it's a string
            }
        
        return team_stats
    
    def _load_predictor(self):
        """Load prediction models."""
        try:
            from src.prediction.predict import GoalPredictor
            predictor = GoalPredictor()
            print("Loaded ML prediction models (XGBoost, CatBoost, Poisson, Dixon-Coles)")
            return predictor
        except Exception as e:
            print(f"Warning: Could not load ML models ({e}). Using simple ELO predictor.")
            return None
    
    def predict_match(self, home_team: str, away_team: str) -> Tuple[int, int, Dict]:
        """
        Predict match outcome using loaded models.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Tuple of (home_goals, away_goals, details_dict)
        """
        # Get team stats
        home_stats = self.team_stats.get(home_team, {'elo': 1500, 'form': 0.5, 'confederation': 'Unknown'})
        away_stats = self.team_stats.get(away_team, {'elo': 1500, 'form': 0.5, 'confederation': 'Unknown'})
        
        # Use ML predictor if available
        if self.predictor is not None:
            return self._predict_with_ml(home_team, away_team, home_stats, away_stats)
        else:
            return self._predict_with_elo(home_team, away_team, home_stats, away_stats)
    
    def _predict_with_ml(self, home_team: str, away_team: str,
                        home_stats: Dict, away_stats: Dict) -> Tuple[int, int, Dict]:
        """Predict using ML models."""
        # Prepare features - ensure confederations are strings (CatBoost requirement)
        home_conf = str(home_stats.get('confederation', 'Unknown'))
        away_conf = str(away_stats.get('confederation', 'Unknown'))
        
        # Handle any remaining NaN values
        if pd.isna(home_conf) or home_conf == 'nan':
            home_conf = 'Unknown'
        if pd.isna(away_conf) or away_conf == 'nan':
            away_conf = 'Unknown'
        
        features = {
            'home_elo_before': home_stats['elo'],
            'away_elo_before': away_stats['elo'],
            'home_form_before': home_stats['form'],
            'away_form_before': away_stats['form'],
            'elo_diff': home_stats['elo'] - away_stats['elo'],
            'form_diff': home_stats['form'] - away_stats['form'],
            'neutral_site': 0,  # World Cup matches are neutral
            'competition': 'FIFA World Cup',
            'home_confederation': home_conf,
            'away_confederation': away_conf,
            # Dummy rolling stats (would be calculated from recent matches in production)
            'home_attack_5': 1.5,
            'away_attack_5': 1.5,
            'home_defense_5': 0.8,
            'away_defense_5': 0.8,
            'home_attack_10': 1.5,
            'away_attack_10': 1.5,
            'home_defense_10': 0.8,
            'away_defense_10': 0.8,
            'home_win_rate_5': 0.5,
            'away_win_rate_5': 0.5,
            'home_win_rate_10': 0.5,
            'away_win_rate_10': 0.5,
            'home_goal_diff_5': 0.5,
            'away_goal_diff_5': 0.5,
            'home_goal_diff_10': 0.5,
            'away_goal_diff_10': 0.5
        }
        
        features_df = pd.DataFrame([features])
        
        # Get predictions
        if self.use_dixon_coles:
            try:
                dc_result = self.predictor.predict_with_dixon_coles(features_df, self.base_model)
                home_expected = dc_result['dc_home'][0]
                away_expected = dc_result['dc_away'][0]
                
                details = {
                    'model': f'dixon_coles_{self.base_model}',
                    'home_expected': home_expected,
                    'away_expected': away_expected,
                    'home_win_prob': dc_result['home_win_prob'][0],
                    'draw_prob': dc_result['draw_prob'][0],
                    'away_win_prob': dc_result['away_win_prob'][0]
                }
            except:
                # Fallback to base model
                result = self.predictor.predict_all(features_df)
                home_expected = result['ensemble_home'].iloc[0]
                away_expected = result['ensemble_away'].iloc[0]
                
                details = {
                    'model': 'ensemble',
                    'home_expected': home_expected,
                    'away_expected': away_expected
                }
        else:
            result = self.predictor.predict_all(features_df)
            home_expected = result[f'{self.base_model}_home'].iloc[0] if f'{self.base_model}_home' in result else result['ensemble_home'].iloc[0]
            away_expected = result[f'{self.base_model}_away'].iloc[0] if f'{self.base_model}_away' in result else result['ensemble_away'].iloc[0]
            
            details = {
                'model': self.base_model,
                'home_expected': home_expected,
                'away_expected': away_expected
            }
        
        # Sample from Poisson distribution
        home_goals = int(np.random.poisson(max(0.01, home_expected)))
        away_goals = int(np.random.poisson(max(0.01, away_expected)))
        
        return home_goals, away_goals, details
    
    def _predict_with_elo(self, home_team: str, away_team: str,
                         home_stats: Dict, away_stats: Dict) -> Tuple[int, int, Dict]:
        """Fallback ELO-based predictor."""
        home_elo = home_stats['elo']
        away_elo = away_stats['elo']
        
        # World Cup is neutral venue, but we can add slight form advantage
        home_elo_adj = home_elo + (home_stats['form'] - 0.5) * 50
        away_elo_adj = away_elo + (away_stats['form'] - 0.5) * 50
        
        # Expected goals based on ELO difference
        elo_diff = home_elo_adj - away_elo_adj
        
        # Convert ELO diff to expected goals
        home_expected = 1.4 + (elo_diff / 400)
        away_expected = 1.4 - (elo_diff / 400)
        
        # Ensure non-negative
        home_expected = max(0.3, min(4.0, home_expected))
        away_expected = max(0.3, min(4.0, away_expected))
        
        # Sample from Poisson distribution
        home_goals = int(np.random.poisson(home_expected))
        away_goals = int(np.random.poisson(away_expected))
        
        details = {
            'model': 'elo_based',
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_expected': home_expected,
            'away_expected': away_expected
        }
        
        return home_goals, away_goals, details


def create_predictor_from_models(use_dixon_coles: bool = True, base_model: str = 'ensemble'):
    """
    Factory function to create a predictor for the tournament simulator.
    
    Args:
        use_dixon_coles: Whether to use Dixon-Coles adjustments
        base_model: Base model to use
        
    Returns:
        Predictor function compatible with TournamentSimulator
    """
    tournament_predictor = TournamentPredictor(use_dixon_coles, base_model)
    
    return tournament_predictor.predict_match
