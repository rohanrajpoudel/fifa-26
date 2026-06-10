"""
Example: Predict a single match using all available models
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.predict import GoalPredictor


def predict_world_cup_final():
    """
    Example prediction for a hypothetical World Cup final.
    """
    print("="*70)
    print("Match Prediction Example: World Cup Final")
    print("="*70)
    
    # Initialize predictor
    predictor = GoalPredictor()
    
    # Load current team data
    teams_df = pd.read_csv('./data/teams_with_elo.csv')
    
    # Get Brazil and Argentina data
    brazil = teams_df[teams_df['team'] == 'Brazil'].iloc[0]
    argentina = teams_df[teams_df['team'] == 'Argentina'].iloc[0]
    
    print(f"\nHome: {brazil['team']}")
    print(f"  ELO: {brazil['current_elo']:.0f}")
    print(f"  Form: {brazil['form']:.3f}")
    
    print(f"\nAway: {argentina['team']}")
    print(f"  ELO: {argentina['current_elo']:.0f}")
    print(f"  Form: {argentina['form']:.3f}")
    
    # Prepare match features
    # Note: In real usage, you'd calculate rolling stats from recent matches
    match_features = {
        'home_elo_before': brazil['current_elo'],
        'away_elo_before': argentina['current_elo'],
        'home_form_before': brazil['form'],
        'away_form_before': argentina['form'],
        'elo_diff': brazil['current_elo'] - argentina['current_elo'],
        'form_diff': brazil['form'] - argentina['form'],
        # Example rolling stats (would be calculated from recent matches)
        'home_attack_5': 1.8,
        'away_attack_5': 1.6,
        'home_defense_5': 0.7,
        'away_defense_5': 0.8,
        'home_attack_10': 1.7,
        'away_attack_10': 1.5,
        'home_defense_10': 0.8,
        'away_defense_10': 0.9,
        'home_win_rate_5': 0.6,
        'away_win_rate_5': 0.6,
        'home_win_rate_10': 0.65,
        'away_win_rate_10': 0.60,
        'home_goal_diff_5': 1.0,
        'away_goal_diff_5': 0.8,
        'home_goal_diff_10': 0.9,
        'away_goal_diff_10': 0.7,
        'neutral_site': 1,
        'competition': 'FIFA World Cup',
        'home_confederation': brazil['confederation'],
        'away_confederation': argentina['confederation']
    }
    
    # Get predictions
    result = predictor.predict_match(
        home_team='Brazil',
        away_team='Argentina',
        features=match_features
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    for model_name, preds in result['predictions'].items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Expected Score: {preds['home_goals']:.2f} - {preds['away_goals']:.2f}")
        
        if 'home_win_prob' in preds:
            print(f"  Home Win: {preds['home_win_prob']:.1%}")
            print(f"  Draw:     {preds['draw_prob']:.1%}")
            print(f"  Away Win: {preds['away_win_prob']:.1%}")


def predict_recent_matches():
    """
    Predict recent real matches and compare with actual results.
    """
    print("\n\n" + "="*70)
    print("Recent Match Predictions vs Actual Results")
    print("="*70)
    
    # Initialize predictor
    predictor = GoalPredictor()
    
    # Load recent matches
    df = pd.read_csv('./data/matches_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Add derived features
    if 'elo_diff' not in df.columns:
        df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    if 'form_diff' not in df.columns:
        df['form_diff'] = df['home_form_before'] - df['away_form_before']
    
    # Get 5 recent matches
    recent = df[df['date'] >= '2024-01-01'].head(5)
    
    for idx, match in recent.iterrows():
        print(f"\n{match['date'].strftime('%Y-%m-%d')}: "
              f"{match['home_team']} vs {match['away_team']}")
        print(f"Competition: {match['competition']}")
        print(f"Actual Score: {int(match['home_goals'])} - {int(match['away_goals'])}")
        
        # Get predictions
        result = predictor.predict_match(
            home_team=match['home_team'],
            away_team=match['away_team'],
            features=match.to_frame().T
        )
        
        print("\nPredictions:")
        
        # Show key models
        for model_name in ['ensemble', 'dixon_coles_xgb']:
            if model_name in result['predictions']:
                preds = result['predictions'][model_name]
                print(f"  {model_name:20s}: {preds['home_goals']:.2f} - {preds['away_goals']:.2f}")
        
        print("-" * 70)


def main():
    """Run all examples."""
    try:
        predict_world_cup_final()
    except Exception as e:
        print(f"Error in World Cup final prediction: {e}")
    
    try:
        predict_recent_matches()
    except Exception as e:
        print(f"Error in recent match predictions: {e}")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
