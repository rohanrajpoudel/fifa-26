"""
ELO Rating System for FIFA Matches
Calculates ELO ratings and form for all teams based on historical match data.
Generates datasets with ELO features for modeling.
"""

import pandas as pd
from collections import defaultdict


WORLD_CUP_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
    "Colombia", "Croatia", "Curacao", "Czech Republic", "DR Congo",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
    "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
]

INITIAL_ELO = 1500
HOME_ADVANTAGE = 30
FORM_WINDOW = 10


def get_k_factor(competition: str):
    """Determine K-factor based on competition importance."""
    c = str(competition).lower()
    
    if "fifa world cup" in c and "qualification" not in c:
        return 60
    
    continental_tournaments = [
        "uefa euro", "copa américa", "copa america",
        "african cup of nations", "afc asian cup",
        "gold cup", "confederations cup", "oceania nations cup"
    ]
    if any(x in c for x in continental_tournaments) and "qualification" not in c:
        return 40
    
    if "nations league" in c and "qualification" not in c:
        return 35
    
    if "qualification" in c or "qualifying" in c:
        return 30
    
    return 20


def compute_form(team, history):
    """Calculate team form based on recent match results."""
    recent = history[-FORM_WINDOW:]
    
    if not recent:
        return 0.5
    
    points = sum(recent)
    return round(points / (3 * len(recent)), 3)


def analyze_competitions(matches_df):
    """Analyze competition categories for K-factor assignment."""
    competitions = sorted(matches_df['competition'].unique())
    
    comp_categories = {
        'World Cup': [],
        'Continental Cup': [],
        'Nations League': [],
        'Qualifiers': [],
        'Friendly': []
    }
    
    for comp in competitions:
        c_lower = comp.lower()
        if 'fifa world cup' in c_lower and 'qualification' not in c_lower:
            comp_categories['World Cup'].append(comp)
        elif any(x in c_lower for x in ['euro', 'copa américa', 'copa america',
                                          'african cup of nations', 'afc asian cup',
                                          'gold cup', 'confederations cup',
                                          'oceania nations cup']) and 'qualification' not in c_lower:
            comp_categories['Continental Cup'].append(comp)
        elif 'nations league' in c_lower and 'qualification' not in c_lower:
            comp_categories['Nations League'].append(comp)
        elif 'qualification' in c_lower or 'qualifying' in c_lower:
            comp_categories['Qualifiers'].append(comp)
        else:
            comp_categories['Friendly'].append(comp)
    
    print("Competition Categories:")
    print(f"\nWorld Cup (K=60): {len(comp_categories['World Cup'])}")
    print(f"Continental Cup (K=40): {len(comp_categories['Continental Cup'])}")
    print(f"Nations League (K=35): {len(comp_categories['Nations League'])}")
    print(f"Qualifiers (K=30): {len(comp_categories['Qualifiers'])}")
    print(f"Friendly (K=20): {len(comp_categories['Friendly'])}")


def calculate_elo_ratings(matches):
    """Calculate ELO ratings for all teams based on match history."""
    ratings = defaultdict(lambda: INITIAL_ELO)
    match_history = defaultdict(list)
    elo_history = []
    feature_rows = []
    
    all_teams = set(matches["home_team"]).union(set(matches["away_team"]))
    
    for i, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        competition = match["competition"]
        neutral = match["neutral_site"]
        
        home_elo_before = ratings[home]
        away_elo_before = ratings[away]
        home_form_before = compute_form(home, match_history[home])
        away_form_before = compute_form(away, match_history[away])
        
        home_adv = 0 if neutral else HOME_ADVANTAGE
        
        expected_home = 1 / (1 + 10 ** ((away_elo_before - home_elo_before - home_adv) / 400))
        expected_away = 1 - expected_home
        
        if home_goals > away_goals:
            actual_home, actual_away = 1, 0
            home_points, away_points = 3, 0
        elif home_goals < away_goals:
            actual_home, actual_away = 0, 1
            home_points, away_points = 0, 3
        else:
            actual_home, actual_away = 0.5, 0.5
            home_points, away_points = 1, 1
        
        K = get_k_factor(competition)
        goal_diff = abs(home_goals - away_goals)
        
        if goal_diff <= 1:
            margin = 1.0
        elif goal_diff == 2:
            margin = 1.3
        else:
            margin = min(1.6 + 0.1 * (goal_diff - 3), 2.0)
        
        ratings[home] += K * margin * (actual_home - expected_home)
        ratings[away] += K * margin * (actual_away - expected_away)
        
        match_history[home].append(home_points)
        match_history[away].append(away_points)
        
        elo_history.append({
            "match_id": i,
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "home_elo_before": home_elo_before,
            "away_elo_before": away_elo_before,
            "home_elo_after": ratings[home],
            "away_elo_after": ratings[away],
        })
        
        feature_rows.append({
            "match_id": i,
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "competition": competition,
            "neutral_site": neutral,
            "home_elo_before": home_elo_before,
            "away_elo_before": away_elo_before,
            "home_form_before": home_form_before,
            "away_form_before": away_form_before,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": (
                "home_win" if home_goals > away_goals else
                "away_win" if home_goals < away_goals else
                "draw"
            )
        })
    
    return ratings, match_history, all_teams, elo_history, feature_rows


def save_datasets(ratings, match_history, all_teams, elo_history, feature_rows):
    """Save all generated datasets to CSV files."""
    teams_df = pd.DataFrame([
        {
            "team": team,
            "current_elo": round(ratings[team], 2),
            "current_form": compute_form(team, match_history[team]),
            "confederation": ""
        }
        for team in sorted(all_teams)
    ])
    
    teams_df.to_csv("./data/teams_with_elo.csv", index=False)
    
    pd.DataFrame(feature_rows).to_csv("./data/matches_with_features.csv", index=False)
    pd.DataFrame(elo_history).to_csv("./data/elo_history.csv", index=False)
    
    wc_2026_teams = teams_df[teams_df["team"].isin(WORLD_CUP_2026)].copy()
    wc_2026_teams = wc_2026_teams.sort_values("current_elo", ascending=False).reset_index(drop=True)
    wc_2026_teams["rank"] = range(1, len(wc_2026_teams) + 1)
    wc_2026_teams = wc_2026_teams[["rank", "team", "current_elo", "current_form"]]
    # wc_2026_teams.to_csv("./data/world_cup_2026_elo.csv", index=False)
    
    wc_matches = pd.DataFrame(feature_rows)
    wc_matches_filtered = wc_matches[
        (wc_matches["home_team"].isin(WORLD_CUP_2026)) |
        (wc_matches["away_team"].isin(WORLD_CUP_2026))
    ].copy()
    # wc_matches_filtered.to_csv("./data/wc_2026_matches_features.csv", index=False)
    
    wc_elo_history = pd.DataFrame(elo_history)
    wc_elo_history_filtered = wc_elo_history[
        (wc_elo_history["home_team"].isin(WORLD_CUP_2026)) |
        (wc_elo_history["away_team"].isin(WORLD_CUP_2026))
    ].copy()
    # wc_elo_history_filtered.to_csv("./data/wc_2026_elo_history.csv", index=False)
    
    return teams_df, wc_2026_teams, wc_matches_filtered


def main():
    """Main execution function."""
    print("Loading cleaned matches...")
    matches = pd.read_csv("./data/cleaned_matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)
    
    print("\nAnalyzing competitions...")
    analyze_competitions(matches)
    
    print("\nCalculating ELO ratings...")
    ratings, match_history, all_teams, elo_history, feature_rows = calculate_elo_ratings(matches)
    
    print("\nSaving datasets...")
    teams_df, wc_2026_teams, wc_matches_filtered = save_datasets(
        ratings, match_history, all_teams, elo_history, feature_rows
    )
    
    print(f"\nTotal matches processed: {len(matches)}")
    print(f"Total teams: {len(all_teams)}")
    print(f"WC 2026 teams: {len(wc_2026_teams)}")
    print(f"WC 2026 matches: {len(wc_matches_filtered)}")
    print(f"\nTop 10 WC 2026 teams by ELO:")
    print(wc_2026_teams.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
