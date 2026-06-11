"""
World Cup 2026 Group Stage Simulator
Handles 12 groups of 4 teams with qualification logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


# Official 2026 World Cup Groups
WORLD_CUP_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}


@dataclass
class TeamStats:
    """Statistics for a team in the group stage."""
    team: str
    group: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    
    @property
    def goal_difference(self):
        return self.goals_for - self.goals_against
    
    def add_result(self, goals_for: int, goals_against: int):
        """Update stats after a match."""
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        
        if goals_for > goals_against:
            self.wins += 1
            self.points += 3
        elif goals_for == goals_against:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1


class GroupStageSimulator:
    """
    Simulates the 2026 World Cup group stage.
    12 groups of 4 teams, round-robin format.
    """
    
    def __init__(self, match_predictor):
        """
        Initialize group stage simulator.
        
        Args:
            match_predictor: Function that predicts match outcomes
                             Should return (home_goals, away_goals)
        """
        self.match_predictor = match_predictor
        self.groups = WORLD_CUP_GROUPS
        
    def simulate_match(self, home_team: str, away_team: str, detailed: bool = False) -> Tuple[int, int, Dict]:
        """
        Simulate a single match between two teams.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            detailed: Whether to return detailed prediction info
            
        Returns:
            Tuple of (home_goals, away_goals, details_dict)
        """
        home_goals, away_goals, details = self.match_predictor(home_team, away_team)
        
        if detailed:
            return home_goals, away_goals, details
        else:
            return home_goals, away_goals, {}
    
    def simulate_group(self, group_name: str, teams: List[str], detailed: bool = False) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Simulate all matches in a group (round-robin).
        
        Args:
            group_name: Group identifier (A-L)
            teams: List of 4 team names
            detailed: Whether to track match details
            
        Returns:
            Tuple of (standings DataFrame, match_results list)
        """
        # Initialize team statistics
        team_stats = {team: TeamStats(team=team, group=group_name) for team in teams}
        match_results = []
        
        # Generate all match pairings (round-robin)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home_team = teams[i]
                away_team = teams[j]
                
                # Simulate match
                home_goals, away_goals, details = self.simulate_match(
                    home_team, away_team, detailed=detailed
                )
                
                # Update statistics
                team_stats[home_team].add_result(home_goals, away_goals)
                team_stats[away_team].add_result(away_goals, home_goals)
                
                # Record match result
                if detailed:
                    match_results.append({
                        'group': group_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'details': details
                    })
        
        # Create standings DataFrame
        standings_data = []
        for team_name, stats in team_stats.items():
            standings_data.append({
                'group': group_name,
                'team': team_name,
                'played': stats.played,
                'wins': stats.wins,
                'draws': stats.draws,
                'losses': stats.losses,
                'goals_for': stats.goals_for,
                'goals_against': stats.goals_against,
                'goal_difference': stats.goal_difference,
                'points': stats.points
            })
        
        standings = pd.DataFrame(standings_data)
        
        # Sort by: points desc, goal difference desc, goals scored desc
        standings = standings.sort_values(
            by=['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        # Add position
        standings.insert(0, 'position', range(1, len(standings) + 1))
        
        return standings, match_results
    
    def simulate_all_groups(self, detailed: bool = False) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
        """
        Simulate all 12 groups in the tournament.
        
        Args:
            detailed: Whether to track detailed match results
            
        Returns:
            Tuple of (group_standings dict, all_match_results list)
        """
        all_standings = {}
        all_matches = []
        
        for group_name, teams in self.groups.items():
            standings, matches = self.simulate_group(group_name, teams, detailed=detailed)
            all_standings[group_name] = standings
            all_matches.extend(matches)
        
        return all_standings, all_matches
    
    def get_qualified_teams(self, all_standings: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Determine which teams qualify from the group stage.
        
        Rules:
        - Top 2 from each group (24 teams)
        - Best 8 third-place teams (8 teams)
        - Total: 32 teams advance to knockout stage
        
        Args:
            all_standings: Dictionary of group standings DataFrames
            
        Returns:
            Dictionary with 'winners', 'runners_up', 'third_place' lists
        """
        winners = []
        runners_up = []
        third_place_teams = []
        
        # Extract top 3 from each group
        for group_name, standings in all_standings.items():
            winners.append(standings.iloc[0]['team'])
            runners_up.append(standings.iloc[1]['team'])
            
            third_place_teams.append({
                'team': standings.iloc[2]['team'],
                'group': group_name,
                'points': standings.iloc[2]['points'],
                'goal_difference': standings.iloc[2]['goal_difference'],
                'goals_for': standings.iloc[2]['goals_for']
            })
        
        # Rank third-place teams
        third_place_df = pd.DataFrame(third_place_teams)
        third_place_df = third_place_df.sort_values(
            by=['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        # Best 8 third-place teams qualify
        best_third_place = third_place_df.head(8)['team'].tolist()
        
        return {
            'winners': winners,
            'runners_up': runners_up,
            'third_place': best_third_place,
            'third_place_full': third_place_df
        }
    
    def print_group_standings(self, all_standings: Dict[str, pd.DataFrame]):
        """Print formatted group standings."""
        for group_name in sorted(all_standings.keys()):
            standings = all_standings[group_name]
            print(f"\n{'='*70}")
            print(f"GROUP {group_name}")
            print('='*70)
            print(f"{'Pos':<4} {'Team':<20} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
            print('-'*70)
            
            for _, row in standings.iterrows():
                print(f"{row['position']:<4} {row['team']:<20} "
                      f"{row['played']:>3} {row['wins']:>3} {row['draws']:>3} {row['losses']:>3} "
                      f"{row['goals_for']:>3} {row['goals_against']:>3} "
                      f"{row['goal_difference']:>4} {row['points']:>4}")
    
    def print_qualified_teams(self, qualified: Dict[str, List[str]]):
        """Print formatted list of qualified teams."""
        print("\n" + "="*70)
        print("QUALIFIED TEAMS FOR ROUND OF 32")
        print("="*70)
        
        print("\nGroup Winners (12 teams):")
        for i, team in enumerate(qualified['winners'], 1):
            print(f"  {i:2}. {team}")
        
        print("\nGroup Runners-up (12 teams):")
        for i, team in enumerate(qualified['runners_up'], 1):
            print(f"  {i:2}. {team}")
        
        print("\nBest Third-Place Teams (8 teams):")
        for i, team in enumerate(qualified['third_place'], 1):
            print(f"  {i:2}. {team}")
        
        print(f"\nTotal: {len(qualified['winners']) + len(qualified['runners_up']) + len(qualified['third_place'])} teams")
