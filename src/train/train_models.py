"""
Train Machine Learning Models for FIFA Match Prediction
Trains XGBoost and CatBoost models to predict home and away goals.
Uses ELO ratings, form, and rolling statistics as features.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load matches data and prepare initial features."""
    df = pd.read_csv('./data/matches_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    df['neutral_site'] = df['neutral_site'].astype(int)
    
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def add_confederation_features(df):
    """Add confederation information for teams."""
    teams_df = pd.read_csv('./data/teams_with_elo.csv')
    teams_df['confederation'] = teams_df['confederation'].fillna('Unknown')
    team_conf = teams_df.set_index('team')['confederation'].to_dict()
    
    df['home_confederation'] = df['home_team'].map(team_conf).fillna('Unknown')
    df['away_confederation'] = df['away_team'].map(team_conf).fillna('Unknown')
    
    print(f"Confederations: {df['home_confederation'].nunique()} unique")
    return df


def create_derived_features(df):
    """Create ELO and form difference features."""
    df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    df['form_diff'] = df['home_form_before'] - df['away_form_before']
    return df


def create_team_stats(df):
    """Create rolling statistics for each team."""
    home_data = df[['date', 'match_id', 'home_team', 'home_goals', 'away_goals']].copy()
    home_data.columns = ['date', 'match_id', 'team', 'goals_scored', 'goals_conceded']
    
    away_data = df[['date', 'match_id', 'away_team', 'away_goals', 'home_goals']].copy()
    away_data.columns = ['date', 'match_id', 'team', 'goals_scored', 'goals_conceded']
    
    team_stats = pd.concat([home_data, away_data], ignore_index=True)
    team_stats = team_stats.sort_values(['date', 'match_id', 'team']).reset_index(drop=True)
    
    team_stats['win'] = (team_stats['goals_scored'] > team_stats['goals_conceded']).astype(int)
    team_stats['goal_diff'] = team_stats['goals_scored'] - team_stats['goals_conceded']
    
    for window in [5, 10]:
        team_stats[f'attack_{window}'] = (
            team_stats.groupby('team')['goals_scored']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_stats[f'defense_{window}'] = (
            team_stats.groupby('team')['goals_conceded']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_stats[f'win_rate_{window}'] = (
            team_stats.groupby('team')['win']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_stats[f'goal_diff_{window}'] = (
            team_stats.groupby('team')['goal_diff']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    return team_stats


def merge_rolling_features(df, team_stats):
    """Merge rolling statistics back to main dataframe."""
    roll_cols = ['attack_5', 'attack_10', 'defense_5', 'defense_10',
                 'win_rate_5', 'win_rate_10', 'goal_diff_5', 'goal_diff_10']
    
    home_merge = team_stats[['date', 'match_id', 'team'] + roll_cols].copy()
    home_merge = home_merge.rename(columns={'team': 'home_team'})
    for col in roll_cols:
        home_merge.rename(columns={col: f'home_{col}'}, inplace=True)
    
    away_merge = team_stats[['date', 'match_id', 'team'] + roll_cols].copy()
    away_merge = away_merge.rename(columns={'team': 'away_team'})
    for col in roll_cols:
        away_merge.rename(columns={col: f'away_{col}'}, inplace=True)
    
    df = df.merge(home_merge, on=['date', 'match_id', 'home_team'], how='left')
    df = df.merge(away_merge, on=['date', 'match_id', 'away_team'], how='left')
    
    df = df.dropna()
    print(f"Matches after feature engineering: {len(df)}")
    
    return df


def split_data(df):
    """Split data into train, validation, and test sets."""
    train = df[df['date'] < '2019-01-01']
    val = df[(df['date'] >= '2019-01-01') & (df['date'] < '2023-01-01')]
    test = df[df['date'] >= '2023-01-01']
    
    print(f"Train: {len(train)} matches ({train['date'].min()} to {train['date'].max()})")
    print(f"Val: {len(val)} matches ({val['date'].min()} to {val['date'].max()})")
    print(f"Test: {len(test)} matches ({test['date'].min()} to {test['date'].max()})")
    
    return train, val, test


def prepare_features(train, val, test):
    """Prepare feature matrices and target variables."""
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
    
    X_train = train[features + cat_features]
    X_val = val[features + cat_features]
    X_test = test[features + cat_features]
    
    y_train_home = train['home_goals']
    y_train_away = train['away_goals']
    y_val_home = val['home_goals']
    y_val_away = val['away_goals']
    y_test_home = test['home_goals']
    y_test_away = test['away_goals']
    
    X_train_encoded = pd.get_dummies(X_train, columns=cat_features, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_features, drop_first=True)
    
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    print(f"Features: {X_train_encoded.shape[1]}")
    
    return (X_train, X_val, X_test, X_train_encoded, X_val_encoded, X_test_encoded,
            y_train_home, y_train_away, y_val_home, y_val_away, y_test_home, y_test_away,
            cat_features)


def train_xgboost_models(X_train_encoded, X_val_encoded, X_test_encoded,
                         y_train_home, y_train_away, y_val_home, y_val_away,
                         y_test_home, y_test_away):
    """Train XGBoost models for home and away goals."""
    xgb_home = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='count:poisson',
        random_state=42
    )
    
    xgb_away = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='count:poisson',
        random_state=42
    )
    
    print("Training XGBoost models...")
    xgb_home.fit(X_train_encoded, y_train_home, eval_set=[(X_val_encoded, y_val_home)], verbose=False)
    xgb_away.fit(X_train_encoded, y_train_away, eval_set=[(X_val_encoded, y_val_away)], verbose=False)
    
    pred_home = np.clip(xgb_home.predict(X_test_encoded), 0.01, None)
    pred_away = np.clip(xgb_away.predict(X_test_encoded), 0.01, None)
    
    print("\nXGBoost Results:")
    print(f"Home - MAE: {mean_absolute_error(y_test_home, pred_home):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_home, pred_home)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_home, pred_home):.4f}")
    print(f"Away - MAE: {mean_absolute_error(y_test_away, pred_away):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_away, pred_away)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_away, pred_away):.4f}")
    
    return xgb_home, xgb_away


def train_catboost_models(X_train, X_val, X_test, y_train_home, y_train_away,
                          y_val_home, y_val_away, y_test_home, y_test_away,
                          cat_features):
    """Train CatBoost models for home and away goals."""
    cat_home = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        loss_function='Poisson',
        random_state=42,
        verbose=False
    )
    
    cat_away = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        loss_function='Poisson',
        random_state=42,
        verbose=False
    )
    
    print("\nTraining CatBoost models...")
    cat_home.fit(X_train, y_train_home, cat_features=cat_features, eval_set=(X_val, y_val_home))
    cat_away.fit(X_train, y_train_away, cat_features=cat_features, eval_set=(X_val, y_val_away))
    
    pred_home = np.clip(cat_home.predict(X_test), 0.01, None)
    pred_away = np.clip(cat_away.predict(X_test), 0.01, None)
    
    print("\nCatBoost Results:")
    print(f"Home - MAE: {mean_absolute_error(y_test_home, pred_home):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_home, pred_home)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_home, pred_home):.4f}")
    print(f"Away - MAE: {mean_absolute_error(y_test_away, pred_away):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_away, pred_away)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_away, pred_away):.4f}")
    
    return cat_home, cat_away


def evaluate_ensemble(xgb_home, xgb_away, cat_home, cat_away,
                     X_test, X_test_encoded, y_test_home, y_test_away):
    """Evaluate ensemble of XGBoost and CatBoost models."""
    xgb_pred_home = np.clip(xgb_home.predict(X_test_encoded), 0.01, None)
    xgb_pred_away = np.clip(xgb_away.predict(X_test_encoded), 0.01, None)
    cat_pred_home = np.clip(cat_home.predict(X_test), 0.01, None)
    cat_pred_away = np.clip(cat_away.predict(X_test), 0.01, None)
    
    ensemble_pred_home = (xgb_pred_home + cat_pred_home) / 2
    ensemble_pred_away = (xgb_pred_away + cat_pred_away) / 2
    
    print("\nEnsemble Results:")
    print(f"Home - MAE: {mean_absolute_error(y_test_home, ensemble_pred_home):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_home, ensemble_pred_home)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_home, ensemble_pred_home):.4f}")
    print(f"Away - MAE: {mean_absolute_error(y_test_away, ensemble_pred_away):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test_away, ensemble_pred_away)):.4f}, "
          f"Poisson: {mean_poisson_deviance(y_test_away, ensemble_pred_away):.4f}")


def save_models(xgb_home, xgb_away, cat_home, cat_away, feature_columns, cat_features):
    """Save trained models and feature information."""
    joblib.dump(xgb_home, './models/xgb_home.pkl')
    joblib.dump(xgb_away, './models/xgb_away.pkl')
    joblib.dump(cat_home, './models/cat_home.pkl')
    joblib.dump(cat_away, './models/cat_away.pkl')
    joblib.dump(feature_columns, './models/feature_columns.pkl')
    joblib.dump(cat_features, './models/cat_features.pkl')
    print("\nModels saved")


def save_validation_predictions(xgb_home, xgb_away, cat_home, cat_away,
                                X_val, X_val_encoded, y_val_home, y_val_away):
    """Save validation predictions for Dixon-Coles rho estimation."""
    val_predictions = pd.DataFrame({
        'actual_home': y_val_home.values,
        'actual_away': y_val_away.values,
        'pred_home_xgb': np.clip(xgb_home.predict(X_val_encoded), 0.01, None),
        'pred_away_xgb': np.clip(xgb_away.predict(X_val_encoded), 0.01, None),
        'pred_home_cat': np.clip(cat_home.predict(X_val), 0.01, None),
        'pred_away_cat': np.clip(cat_away.predict(X_val), 0.01, None)
    })
    
    val_predictions.to_csv('./models/results/validation_predictions.csv', index=False)
    print("Validation predictions saved for Dixon-Coles rho estimation")


def main():
    """Main execution function."""
    df = load_and_prepare_data()
    df = add_confederation_features(df)
    df = create_derived_features(df)
    
    print("\nCreating rolling features...")
    team_stats = create_team_stats(df)
    print(f"Team stats rows: {len(team_stats)}")
    
    df = merge_rolling_features(df, team_stats)
    
    train, val, test = split_data(df)
    
    (X_train, X_val, X_test, X_train_encoded, X_val_encoded, X_test_encoded,
     y_train_home, y_train_away, y_val_home, y_val_away, y_test_home, y_test_away,
     cat_features) = prepare_features(train, val, test)
    
    xgb_home, xgb_away = train_xgboost_models(
        X_train_encoded, X_val_encoded, X_test_encoded,
        y_train_home, y_train_away, y_val_home, y_val_away,
        y_test_home, y_test_away
    )
    
    cat_home, cat_away = train_catboost_models(
        X_train, X_val, X_test,
        y_train_home, y_train_away, y_val_home, y_val_away,
        y_test_home, y_test_away, cat_features
    )
    
    evaluate_ensemble(xgb_home, xgb_away, cat_home, cat_away,
                     X_test, X_test_encoded, y_test_home, y_test_away)
    
    save_models(xgb_home, xgb_away, cat_home, cat_away,
                X_train_encoded.columns.tolist(), cat_features)
    
    save_validation_predictions(xgb_home, xgb_away, cat_home, cat_away,
                               X_val, X_val_encoded, y_val_home, y_val_away)


if __name__ == "__main__":
    main()
