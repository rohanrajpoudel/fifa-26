"""
Quick test of the simulation system.
Tests basic functionality without ML models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.tournament import TournamentSimulator, create_simple_predictor
import pandas as pd


def test_simulation():
    """Test basic simulation functionality."""
    print("="*70)
    print("TESTING WORLD CUP 2026 SIMULATOR")
    print("="*70)
    
    # Load team data
    print("\n1. Loading team data...")
    try:
        teams_df = pd.read_csv('./data/teams_with_elo.csv')
        print(f"   [OK] Loaded {len(teams_df)} teams")
    except FileNotFoundError:
        print("   [ERROR] teams_with_elo.csv not found")
        print("   Run: python src/elo/generate_elo.py first")
        return False
    
    # Create simple predictor
    print("\n2. Creating predictor...")
    predictor = create_simple_predictor(teams_df)
    print("   [OK] Simple ELO-based predictor created")
    
    # Test single match prediction
    print("\n3. Testing match prediction...")
    home_goals, away_goals, details = predictor("Brazil", "Argentina")
    print(f"   Brazil vs Argentina: {home_goals}-{away_goals}")
    print(f"   Expected: Home={details['home_expected']:.2f}, Away={details['away_expected']:.2f}")
    
    # Create simulator
    print("\n4. Creating tournament simulator...")
    simulator = TournamentSimulator(predictor)
    print("   [OK] Simulator initialized")
    
    # Test single tournament
    print("\n5. Running single tournament simulation...")
    result = simulator.simulate_tournament(detailed=False)
    print(f"   [OK] Tournament complete")
    print(f"   Champion: {result['champion']}")
    
    # Test Monte Carlo (small sample)
    print("\n6. Running Monte Carlo simulation (100 iterations)...")
    results_df = simulator.run_monte_carlo(100)
    print(f"   [OK] Monte Carlo complete")
    print(f"\n   Top 5 Favorites:")
    for idx, row in results_df.head(5).iterrows():
        print(f"   {idx+1}. {row['team']:<20} {row['championship_probability']*100:>6.2f}%")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED [SUCCESS]")
    print("="*70)
    print("\nSimulator is ready to use!")
    print("\nNext steps:")
    print("  • Run full simulation: python src/run_simulation.py --monte-carlo 10000")
    print("  • Start web interface: streamlit run src/app/app.py")
    print("  • Train ML models: python src/prediction/train_all.py")
    
    return True


if __name__ == "__main__":
    try:
        success = test_simulation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
