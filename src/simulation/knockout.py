"""
World Cup 2026 Knockout Stage Simulator
Handles Round of 32 through Final with extra time and penalties.
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class KnockoutMatch:
    """Result of a knockout match."""
    round_name: str
    match_id: int
    home_team: str
    away_team: str
    home_goals_90: int
    away_goals_90: int
    went_to_extra_time: bool = False
    home_goals_et: int = 0
    away_goals_et: int = 0
    went_to_penalties: bool = False
    penalty_winner: str = None
    winner: str = None
    
    @property
    def home_goals_total(self):
        return self.home_goals_90 + self.home_goals_et
    
    @property
    def away_goals_total(self):
        return self.away_goals_90 + self.away_goals_et


class KnockoutSimulator:
    """
    Simulates knockout rounds of 2026 World Cup.
    Handles extra time and penalty shootouts.
    """
    
    def __init__(self, match_predictor):
        """
        Initialize knockout simulator.
        
        Args:
            match_predictor: Function that predicts match outcomes
                             Should return (home_goals, away_goals, details)
        """
        self.match_predictor = match_predictor
        
    def simulate_knockout_match(self, home_team: str, away_team: str, 
                                round_name: str, match_id: int,
                                detailed: bool = False) -> KnockoutMatch:
        """
        Simulate a single knockout match with extra time and penalties.
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            round_name: Name of the round (e.g., "Round of 32")
            match_id: Match identifier within the round
            detailed: Whether to track detailed prediction info
            
        Returns:
            KnockoutMatch object with complete results
        """
        # Simulate 90 minutes
        home_goals_90, away_goals_90, details = self.match_predictor(home_team, away_team)
        
        match = KnockoutMatch(
            round_name=round_name,
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_goals_90=home_goals_90,
            away_goals_90=away_goals_90
        )
        
        # Check if match is tied after 90 minutes
        if home_goals_90 == away_goals_90:
            # Extra time (30 minutes)
            match.went_to_extra_time = True
            match.home_goals_et, match.away_goals_et = self._simulate_extra_time(
                home_team, away_team
            )
            
            # Check if still tied after extra time
            if match.home_goals_total == match.away_goals_total:
                # Penalty shootout
                match.went_to_penalties = True
                match.penalty_winner = self._simulate_penalties(home_team, away_team)
                match.winner = match.penalty_winner
            else:
                # Winner decided in extra time
                match.winner = home_team if match.home_goals_total > match.away_goals_total else away_team
        else:
            # Winner decided in 90 minutes
            match.winner = home_team if home_goals_90 > away_goals_90 else away_team
        
        return match
    
    def _simulate_extra_time(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """
        Simulate extra time (30 minutes).
        
        Extra time typically sees fewer goals, so we reduce the expected goals.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Tuple of (home_goals, away_goals) in extra time
        """
        # Get base prediction
        base_home, base_away, _ = self.match_predictor(home_team, away_team)
        
        # Extra time is 1/3 of normal time, and typically more defensive
        # Reduce expected goals by ~60%
        et_home = np.random.poisson(max(0.01, base_home * 0.25))
        et_away = np.random.poisson(max(0.01, base_away * 0.25))
        
        return int(et_home), int(et_away)
    
    def _simulate_penalties(self, home_team: str, away_team: str) -> str:
        """
        Simulate penalty shootout.
        
        Simple model: 50/50 with slight home advantage.
        In reality, could use team quality, pressure handling, etc.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Name of the winning team
        """
        # Slight home advantage in penalties (52-48)
        return home_team if np.random.random() < 0.52 else away_team
    
    def create_round_of_32_bracket(self, qualified_teams: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Create Round of 32 matchups based on qualified teams.
        
        Simplified bracket structure - pairs winners, runners-up, and third-place teams.
        
        Args:
            qualified_teams: Dictionary with winners, runners_up, third_place lists
            
        Returns:
            List of 16 matchup tuples (team1, team2)
        """
        winners = qualified_teams['winners']
        runners_up = qualified_teams['runners_up']
        third_place = qualified_teams['third_place']
        
        matchups = []
        
        # Matchups 1-8: Winners vs Third-place teams (if enough third place)
        for i in range(min(8, len(third_place))):
            matchups.append((winners[i], third_place[i]))
        
        # If fewer than 8 third-place teams, pair remaining winners with runners-up
        if len(third_place) < 8:
            for i in range(len(third_place), 8):
                matchups.append((winners[i], runners_up[i]))
        
        # Matchups 9-16: Remaining winners vs runners-up
        for i in range(8, 12):
            matchups.append((winners[i], runners_up[11 - i]))  # Cross-pairing
        
        # Pair remaining runners-up if needed
        remaining_runners = len(runners_up) - (12 - max(8, len(third_place)))
        if remaining_runners > 0:
            for i in range(remaining_runners // 2):
                matchups.append((runners_up[i], runners_up[remaining_runners - 1 - i]))
        
        return matchups[:16]  # Ensure exactly 16 matches
    
    def simulate_knockout_round(self, round_name: str, matchups: List[Tuple[str, str]], 
                                detailed: bool = False) -> Tuple[List[str], List[KnockoutMatch]]:
        """
        Simulate a complete knockout round.
        
        Args:
            round_name: Name of the round
            matchups: List of (team1, team2) tuples
            detailed: Whether to track detailed results
            
        Returns:
            Tuple of (winners list, match_results list)
        """
        winners = []
        match_results = []
        
        for match_id, (team1, team2) in enumerate(matchups, 1):
            match = self.simulate_knockout_match(
                home_team=team1,
                away_team=team2,
                round_name=round_name,
                match_id=match_id,
                detailed=detailed
            )
            
            winners.append(match.winner)
            
            if detailed:
                match_results.append(match)
        
        return winners, match_results
    
    def simulate_knockout_stage(self, qualified_teams: Dict[str, List[str]], 
                                detailed: bool = False) -> Tuple[str, List[KnockoutMatch]]:
        """
        Simulate entire knockout stage from Round of 32 to Final.
        
        Args:
            qualified_teams: Dictionary with qualified teams from group stage
            detailed: Whether to track detailed results
            
        Returns:
            Tuple of (champion name, all_match_results list)
        """
        all_matches = []
        
        # Round of 32 (32 → 16)
        r32_matchups = self.create_round_of_32_bracket(qualified_teams)
        r16_teams, r32_matches = self.simulate_knockout_round("Round of 32", r32_matchups, detailed)
        all_matches.extend(r32_matches)
        
        # Round of 16 (16 → 8)
        r16_matchups = [(r16_teams[i], r16_teams[i+1]) for i in range(0, len(r16_teams), 2)]
        qf_teams, r16_matches = self.simulate_knockout_round("Round of 16", r16_matchups, detailed)
        all_matches.extend(r16_matches)
        
        # Quarter-finals (8 → 4)
        qf_matchups = [(qf_teams[i], qf_teams[i+1]) for i in range(0, len(qf_teams), 2)]
        sf_teams, qf_matches = self.simulate_knockout_round("Quarter-finals", qf_matchups, detailed)
        all_matches.extend(qf_matches)
        
        # Semi-finals (4 → 2)
        sf_matchups = [(sf_teams[i], sf_teams[i+1]) for i in range(0, len(sf_teams), 2)]
        final_teams, sf_matches = self.simulate_knockout_round("Semi-finals", sf_matchups, detailed)
        all_matches.extend(sf_matches)
        
        # Final (2 → 1)
        final_matchup = [(final_teams[0], final_teams[1])]
        champion, final_match = self.simulate_knockout_round("Final", final_matchup, detailed)
        all_matches.extend(final_match)
        
        return champion[0], all_matches
    
    def print_knockout_match(self, match: KnockoutMatch):
        """Print formatted knockout match result."""
        print(f"\n{match.round_name} - Match {match.match_id}")
        print(f"  {match.home_team} {match.home_goals_90} - {match.away_goals_90} {match.away_team}")
        
        if match.went_to_extra_time:
            print(f"  After Extra Time: {match.home_goals_total} - {match.away_goals_total}")
        
        if match.went_to_penalties:
            print(f"  Penalties: {match.penalty_winner} wins")
        
        print(f"  Winner: {match.winner}")
    
    def print_knockout_results(self, matches: List[KnockoutMatch]):
        """Print all knockout match results."""
        rounds = {}
        for match in matches:
            if match.round_name not in rounds:
                rounds[match.round_name] = []
            rounds[match.round_name].append(match)
        
        round_order = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        
        for round_name in round_order:
            if round_name in rounds:
                print(f"\n{'='*70}")
                print(f"{round_name.upper()}")
                print('='*70)
                
                for match in rounds[round_name]:
                    self.print_knockout_match(match)
