"""
Complete World Cup 2026 Tournament Simulator
Combines group stage and knockout stage with Monte Carlo simulation.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Callable
import time

from .groups import GroupStageSimulator
from .knockout import KnockoutSimulator


class TournamentSimulator:
    """
    Complete 2026 World Cup tournament simulator.
    Runs group stage + knockout stage with Monte Carlo aggregation.
    """
    
    def __init__(self, match_predictor: Callable):
        """
        Initialize tournament simulator.
        
        Args:
            match_predictor: Function that predicts match outcomes
                            Should return (home_goals, away_goals, details_dict)
        """
        self.match_predictor = match_predictor
        self.group_simulator = GroupStageSimulator(match_predictor)
        self.knockout_simulator = KnockoutSimulator(match_predictor)
        
        # Results storage
        self.simulation_results = []
        
    def simulate_tournament(self, detailed: bool = False) -> Dict:
        """
        Simulate one complete World Cup tournament.
        
        Args:
            detailed: Whether to track all match details
            
        Returns:
            Dictionary with tournament results
        """
        # Group stage
        group_standings, group_matches = self.group_simulator.simulate_all_groups(detailed)
        qualified = self.group_simulator.get_qualified_teams(group_standings)
        
        # Knockout stage
        champion, knockout_matches = self.knockout_simulator.simulate_knockout_stage(qualified, detailed)
        
        # Compile results
        results = {
            'champion': champion,
            'group_standings': group_standings,
            'qualified_teams': qualified,
            'group_matches': group_matches if detailed else [],
            'knockout_matches': knockout_matches if detailed else []
        }
        
        return results
    
    def run_monte_carlo(self, n_simulations: int, progress_callback=None) -> pd.DataFrame:
        """
        Run Monte Carlo simulation of the tournament.
        
        Args:
            n_simulations: Number of tournament simulations to run
            progress_callback: Optional callback function for progress updates
                              Should accept (current, total) arguments
            
        Returns:
            DataFrame with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running {n_simulations:,} World Cup 2026 Simulations")
        print('='*70)
        
        champions = []
        group_winners = defaultdict(list)
        group_positions = defaultdict(lambda: defaultdict(int))
        knockout_appearances = defaultdict(int)
        
        start_time = time.time()
        
        for sim_num in range(n_simulations):
            # Run one tournament
            result = self.simulate_tournament(detailed=False)
            
            # Record champion
            champions.append(result['champion'])
            
            # Record group winners
            for group_name, standings in result['group_standings'].items():
                winner = standings.iloc[0]['team']
                group_winners[group_name].append(winner)
                
                # Record all group positions
                for idx, row in standings.iterrows():
                    position = idx + 1
                    team = row['team']
                    group_positions[team][position] += 1
            
            # Record knockout appearances
            all_qualified = (result['qualified_teams']['winners'] + 
                           result['qualified_teams']['runners_up'] + 
                           result['qualified_teams']['third_place'])
            
            for team in all_qualified:
                knockout_appearances[team] += 1
            
            # Progress update
            if progress_callback and (sim_num + 1) % 100 == 0:
                progress_callback(sim_num + 1, n_simulations)
            elif (sim_num + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (sim_num + 1) / elapsed
                remaining = (n_simulations - sim_num - 1) / rate
                print(f"  Progress: {sim_num + 1:,}/{n_simulations:,} "
                      f"({100*(sim_num+1)/n_simulations:.1f}%) "
                      f"- Est. remaining: {remaining:.0f}s")
        
        elapsed_total = time.time() - start_time
        print(f"\nCompleted {n_simulations:,} simulations in {elapsed_total:.1f}s "
              f"({n_simulations/elapsed_total:.0f} sims/sec)")
        
        # Aggregate results
        results_df = self._aggregate_results(
            champions, group_winners, group_positions, 
            knockout_appearances, n_simulations
        )
        
        return results_df
    
    def _aggregate_results(self, champions: List[str], group_winners: Dict,
                          group_positions: Dict, knockout_appearances: Dict,
                          n_simulations: int) -> pd.DataFrame:
        """Aggregate Monte Carlo simulation results."""
        
        # Count champion occurrences
        champion_counts = Counter(champions)
        
        # Prepare results data
        results_data = []
        
        for team in set(champions):
            wins = champion_counts[team]
            win_prob = wins / n_simulations
            knockout_prob = knockout_appearances.get(team, 0) / n_simulations
            
            # Calculate average group position
            positions = group_positions.get(team, {})
            if positions:
                avg_position = sum(pos * count for pos, count in positions.items()) / sum(positions.values())
            else:
                avg_position = None
            
            results_data.append({
                'team': team,
                'championship_wins': wins,
                'championship_probability': win_prob,
                'knockout_appearances': knockout_appearances.get(team, 0),
                'knockout_probability': knockout_prob,
                'avg_group_position': avg_position
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('championship_probability', ascending=False)
        results_df = results_df.reset_index(drop=True)
        
        return results_df
    
    def print_simulation_summary(self, results_df: pd.DataFrame, top_n: int = 20):
        """Print formatted summary of simulation results."""
        print(f"\n{'='*70}")
        print("WORLD CUP 2026 - SIMULATION RESULTS")
        print('='*70)
        
        print(f"\nTop {top_n} Championship Favorites:")
        print(f"{'Rank':<6} {'Team':<25} {'Win %':>10} {'Knockout %':>12} {'Avg Pos':>10}")
        print('-'*70)
        
        for idx, row in results_df.head(top_n).iterrows():
            rank = idx + 1
            print(f"{rank:<6} {row['team']:<25} "
                  f"{row['championship_probability']*100:>9.2f}% "
                  f"{row['knockout_probability']*100:>11.2f}% "
                  f"{row['avg_group_position']:>10.2f}")
        
        print(f"\n{'='*70}")
        print(f"Most Likely Champion: {results_df.iloc[0]['team']} "
              f"({results_df.iloc[0]['championship_probability']*100:.2f}%)")
        print('='*70)
    
    def simulate_single_tournament_detailed(self) -> Dict:
        """
        Simulate one tournament with full details for reporting.
        
        Returns:
            Dictionary with complete tournament information
        """
        print(f"\n{'='*70}")
        print("WORLD CUP 2026 - DETAILED TOURNAMENT SIMULATION")
        print('='*70)
        
        # Group stage
        print("\n" + "="*70)
        print("GROUP STAGE")
        print("="*70)
        
        group_standings, group_matches = self.group_simulator.simulate_all_groups(detailed=True)
        self.group_simulator.print_group_standings(group_standings)
        
        # Qualified teams
        qualified = self.group_simulator.get_qualified_teams(group_standings)
        self.group_simulator.print_qualified_teams(qualified)
        
        # Third-place ranking
        print("\n" + "="*70)
        print("THIRD-PLACE TEAMS RANKING")
        print("="*70)
        third_place_df = qualified['third_place_full']
        print(f"{'Rank':<6} {'Team':<20} {'Group':<8} {'Pts':>4} {'GD':>4} {'GF':>4}")
        print('-'*70)
        for idx, row in third_place_df.iterrows():
            rank = idx + 1
            qualified_mark = "[OK]" if row['team'] in qualified['third_place'] else "[X]"
            print(f"{rank:<6} {row['team']:<20} {row['group']:<8} "
                  f"{row['points']:>4} {row['goal_difference']:>4} {row['goals_for']:>4} {qualified_mark}")
        
        # Knockout stage
        print("\n" + "="*70)
        print("KNOCKOUT STAGE")
        print("="*70)
        
        champion, knockout_matches = self.knockout_simulator.simulate_knockout_stage(qualified, detailed=True)
        self.knockout_simulator.print_knockout_results(knockout_matches)
        
        # Champion announcement
        print("\n" + "="*70)
        print(f"🏆 WORLD CUP 2026 CHAMPION: {champion.upper()} 🏆")
        print("="*70)
        
        return {
            'champion': champion,
            'group_standings': group_standings,
            'group_matches': group_matches,
            'qualified_teams': qualified,
            'knockout_matches': knockout_matches
        }
    
    def export_results(self, results_df: pd.DataFrame, filepath: str = './results/simulation_results.csv'):
        """Export simulation results to CSV."""
        results_df.to_csv(filepath, index=False)
        print(f"\nResults exported to {filepath}")


def create_simple_predictor(teams_df: pd.DataFrame):
    """
    Create a simple match predictor based on ELO ratings.
    
    This is a basic predictor for testing. In production, use the
    full prediction models from src/prediction/.
    
    Args:
        teams_df: DataFrame with team ELO ratings
        
    Returns:
        Predictor function
    """
    team_elo = teams_df.set_index('team')['current_elo'].to_dict()
    
    def predictor(home_team: str, away_team: str) -> Tuple[int, int, Dict]:
        """Simple ELO-based predictor."""
        home_elo = team_elo.get(home_team, 1500)
        away_elo = team_elo.get(away_team, 1500)
        
        # Add home advantage
        home_elo_adj = home_elo + 30
        
        # Expected goals based on ELO difference
        elo_diff = home_elo_adj - away_elo
        
        # Convert ELO diff to expected goals (simplified)
        home_expected = 1.5 + (elo_diff / 400)
        away_expected = 1.2 - (elo_diff / 400)
        
        # Ensure non-negative
        home_expected = max(0.3, home_expected)
        away_expected = max(0.3, away_expected)
        
        # Sample from Poisson distribution
        home_goals = np.random.poisson(home_expected)
        away_goals = np.random.poisson(away_expected)
        
        details = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_expected': home_expected,
            'away_expected': away_expected
        }
        
        return int(home_goals), int(away_goals), details
    
    return predictor
