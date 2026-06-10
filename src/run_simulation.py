"""
Command-line interface for World Cup 2026 simulator.
Run simulations without the Streamlit frontend.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.tournament import TournamentSimulator
from src.simulation.predictor_integration import create_predictor_from_models


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Tournament Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10,000 Monte Carlo simulations
  python run_simulation.py --monte-carlo 10000

  # Run detailed single tournament
  python run_simulation.py --detailed

  # Use specific model
  python run_simulation.py --monte-carlo 5000 --model xgb --no-dixon-coles

  # Save results to custom file
  python run_simulation.py --monte-carlo 10000 --output my_results.csv
        """
    )
    
    # Simulation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--monte-carlo', '-mc',
        type=int,
        metavar='N',
        help='Run N Monte Carlo simulations'
    )
    mode_group.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Run single detailed tournament simulation'
    )
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        choices=['ensemble', 'xgb', 'cat', 'poisson'],
        default='ensemble',
        help='Base prediction model to use (default: ensemble)'
    )
    parser.add_argument(
        '--no-dixon-coles',
        action='store_true',
        help='Disable Dixon-Coles adjustment'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results/simulation_results.csv',
        help='Output file for Monte Carlo results (default: ./results/simulation_results.csv)'
    )
    parser.add_argument(
        '--top', '-t',
        type=int,
        default=20,
        help='Number of top teams to display (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Create results directory
    Path('./results').mkdir(exist_ok=True)
    
    # Configuration
    use_dixon_coles = not args.no_dixon_coles
    base_model = args.model
    
    print(f"\n{'='*70}")
    print("FIFA WORLD CUP 2026 SIMULATOR")
    print('='*70)
    print(f"Base Model: {base_model}")
    print(f"Dixon-Coles: {'Enabled' if use_dixon_coles else 'Disabled'}")
    print('='*70)
    
    try:
        # Create predictor
        print("\nLoading prediction models...")
        predictor = create_predictor_from_models(use_dixon_coles, base_model)
        
        # Create simulator
        simulator = TournamentSimulator(predictor)
        
        if args.detailed:
            # Run detailed single tournament
            result = simulator.simulate_single_tournament_detailed()
            
            print(f"\n{'='*70}")
            print("Simulation complete!")
            print('='*70)
            
        else:
            # Run Monte Carlo simulation
            n_sims = args.monte_carlo
            
            results_df = simulator.run_monte_carlo(n_sims)
            
            # Display summary
            simulator.print_simulation_summary(results_df, top_n=args.top)
            
            # Save results
            results_df.to_csv(args.output, index=False)
            print(f"\n[SUCCESS] Results saved to {args.output}")
            
            # Additional statistics
            print(f"\n{'='*70}")
            print("ADDITIONAL STATISTICS")
            print('='*70)
            
            top_5 = results_df.head(5)
            print(f"\nTop 5 Combined Win Probability: {top_5['championship_probability'].sum()*100:.2f}%")
            
            # Probability of upset (champion outside top 5)
            upset_prob = 1 - top_5['championship_probability'].sum()
            print(f"Probability of Upset (champion not in top 5): {upset_prob*100:.2f}%")
            
            # Most consistent team (high knockout probability)
            most_consistent = results_df.sort_values('knockout_probability', ascending=False).iloc[0]
            print(f"\nMost Consistent Team: {most_consistent['team']} "
                  f"({most_consistent['knockout_probability']*100:.2f}% knockout qualification)")
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
