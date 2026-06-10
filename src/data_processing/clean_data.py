"""
Data Cleaning Script
Processes raw FIFA match data and creates cleaned datasets for matches and teams.
Date range: 2000-01-01 to 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data():
    """Load raw data files and filter by date range."""
    results_df = pd.read_csv('./rawData/results.csv')
    goalscorers_df = pd.read_csv('./rawData/goalscorers.csv')
    shootouts_df = pd.read_csv('./rawData/shootouts.csv')
    former_names_df = pd.read_csv('./rawData/former_names.csv')
    
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df[(results_df['date'] >= '2000-01-01') & (results_df['date'] < '2027-01-01')]
    
    goalscorers_df['date'] = pd.to_datetime(goalscorers_df['date'])
    goalscorers_df = goalscorers_df[(goalscorers_df['date'] >= '2000-01-01') & (goalscorers_df['date'] < '2027-01-01')]
    
    shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])
    shootouts_df = shootouts_df[(shootouts_df['date'] >= '2000-01-01') & (shootouts_df['date'] < '2027-01-01')]
    
    return results_df, goalscorers_df, shootouts_df, former_names_df


def process_matches(results_df, goalscorers_df, shootouts_df):
    """Process matches data with statistics and additional information."""
    matches = results_df.copy()
    matches['year'] = matches['date'].dt.year
    matches['month'] = matches['date'].dt.month
    matches['total_goals'] = matches['home_score'] + matches['away_score']
    
    matches['result'] = 'draw'
    matches.loc[matches['home_score'] > matches['away_score'], 'result'] = 'home_win'
    matches.loc[matches['home_score'] < matches['away_score'], 'result'] = 'away_win'
    
    goal_counts = goalscorers_df.groupby(['date', 'home_team', 'away_team']).agg({
        'scorer': 'count',
        'own_goal': 'sum',
        'penalty': 'sum'
    }).rename(columns={'scorer': 'goal_count', 'own_goal': 'own_goals', 'penalty': 'penalties'})
    
    matches = matches.merge(goal_counts, on=['date', 'home_team', 'away_team'], how='left')
    matches['goal_count'] = matches['goal_count'].fillna(0).astype(int)
    matches['own_goals'] = matches['own_goals'].fillna(0).astype(int)
    matches['penalties'] = matches['penalties'].fillna(0).astype(int)
    
    matches = matches.merge(
        shootouts_df[['date', 'home_team', 'away_team', 'winner']].rename(columns={'winner': 'shootout_winner'}),
        on=['date', 'home_team', 'away_team'],
        how='left'
    )
    matches['had_shootout'] = matches['shootout_winner'].notna()
    
    matches_clean = matches[[
        'date', 'year', 'month', 'home_team', 'away_team',
        'home_score', 'away_score', 'total_goals', 'result',
        'tournament', 'city', 'country', 'neutral',
        'goal_count', 'own_goals', 'penalties', 'had_shootout', 'shootout_winner'
    ]].sort_values('date', ascending=False)
    
    return matches_clean


def process_teams(results_df, former_names_df):
    """Process teams data with comprehensive statistics."""
    all_teams = pd.concat([
        results_df[['home_team']].rename(columns={'home_team': 'team'}),
        results_df[['away_team']].rename(columns={'away_team': 'team'})
    ]).drop_duplicates()
    
    teams_stats = []
    
    for team in all_teams['team'].unique():
        home_matches = results_df[results_df['home_team'] == team]
        away_matches = results_df[results_df['away_team'] == team]
        
        total_matches = len(home_matches) + len(away_matches)
        
        home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
        away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
        total_wins = home_wins + away_wins
        
        home_draws = len(home_matches[home_matches['home_score'] == home_matches['away_score']])
        away_draws = len(away_matches[away_matches['away_score'] == away_matches['home_score']])
        total_draws = home_draws + away_draws
        
        total_losses = total_matches - total_wins - total_draws
        
        goals_scored = home_matches['home_score'].sum() + away_matches['away_score'].sum()
        goals_conceded = home_matches['away_score'].sum() + away_matches['home_score'].sum()
        goal_difference = goals_scored - goals_conceded
        
        win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        
        all_team_matches = pd.concat([home_matches, away_matches])
        first_match = all_team_matches['date'].min()
        last_match = all_team_matches['date'].max()
        
        former_names = former_names_df[former_names_df['current'] == team]['former'].tolist()
        has_former_name = len(former_names) > 0
        former_name_list = ';'.join(former_names) if former_names else None
        
        teams_stats.append({
            'team': team,
            'total_matches': total_matches,
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_losses,
            'goals_scored': int(goals_scored),
            'goals_conceded': int(goals_conceded),
            'goal_difference': int(goal_difference),
            'win_percentage': round(win_percentage, 2),
            'first_match': first_match,
            'last_match': last_match,
            'has_former_name': has_former_name,
            'former_names': former_name_list
        })
    
    teams_clean = pd.DataFrame(teams_stats).sort_values('total_matches', ascending=False)
    return teams_clean


def create_cleaned_matches(matches_df):
    """Create simplified cleaned matches dataset for modeling."""
    df = matches_df[[
        "date", "home_team", "away_team", "tournament",
        "home_score", "away_score", "neutral"
    ]].copy()
    
    df.rename(
        columns={
            "tournament": "competition",
            "home_score": "home_goals",
            "away_score": "away_goals",
            "neutral": "neutral_site"
        },
        inplace=True
    )
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.insert(0, "match_id", range(1, len(df) + 1))
    
    return df


def create_cleaned_teams(matches_df):
    """Create cleaned teams list from matches."""
    teams = sorted(
        set(matches_df["home_team"]).union(set(matches_df["away_team"]))
    )
    teams_df = pd.DataFrame({"team": teams})
    return teams_df


def main():
    """Main execution function."""
    print("Loading raw data...")
    results_df, goalscorers_df, shootouts_df, former_names_df = load_raw_data()
    
    print("Processing matches...")
    matches_clean = process_matches(results_df, goalscorers_df, shootouts_df)
    
    print("Processing teams...")
    teams_clean = process_teams(results_df, former_names_df)
    
    Path('./data').mkdir(parents=True, exist_ok=True)
    
    # matches_clean.to_csv('./data/matches.csv', index=False)
    # teams_clean.to_csv('./data/teams.csv', index=False)
    
    print(f"Matches: {len(matches_clean)} rows saved")
    print(f"Teams: {len(teams_clean)} rows saved")
    
    print("\nMatches sample:")
    print(matches_clean.head())
    print("\nTeams sample:")
    print(teams_clean.head())
    
    print("\nCreating simplified datasets for modeling...")
    cleaned_matches = create_cleaned_matches(matches_clean)
    cleaned_teams = create_cleaned_teams(cleaned_matches)
    
    cleaned_matches.to_csv("./data/cleaned_matches.csv", index=False)
    # cleaned_teams.to_csv("./data/cleaned_teams.csv", index=False)
    
    print(f"Saved {len(cleaned_matches)} cleaned matches")
    print(f"Generated {len(cleaned_teams)} cleaned teams")


if __name__ == "__main__":
    main()
