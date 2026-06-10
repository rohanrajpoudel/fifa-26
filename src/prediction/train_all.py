"""
Complete Training Pipeline for Goal Prediction Models
Trains Poisson Regression and Dixon-Coles models using existing XGBoost/CatBoost as base.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.prediction.poisson_model import train_poisson_model
from src.prediction.dixon_coles import train_dixon_coles_with_xgboost, train_dixon_coles_with_catboost
from src.prediction.evaluate_models import ModelEvaluator


def main():
    """
    Complete training and evaluation pipeline.
    
    Steps:
    1. Train Poisson regression model
    2. Train Dixon-Coles adjustments for XGBoost
    3. Train Dixon-Coles adjustments for CatBoost
    4. Evaluate all models on test set
    5. Generate comparison report
    """
    
    print("="*70)
    print("FIFA GOAL PREDICTION TRAINING PIPELINE")
    print("="*70)
    print("\nPhase 3: Moving from Win/Draw/Loss to Expected Goals")
    print("Methods: Poisson Regression + Dixon-Coles Model")
    print("="*70)
    
    # Step 1: Train Poisson Regression
    print("\n\n[1/4] Training Poisson Regression Model")
    print("-"*70)
    try:
        poisson_model = train_poisson_model('./data/matches_with_features.csv')
        poisson_model.save('./models/poisson_model.pkl')
        print("\nPoisson model trained and saved successfully")
    except Exception as e:
        print(f"\nError training Poisson model: {e}")
        print("Continuing with Dixon-Coles training...")
    
    # Step 2: Train Dixon-Coles with XGBoost
    print("\n\n[2/4] Training Dixon-Coles with XGBoost Base")
    print("-"*70)
    try:
        dc_xgb = train_dixon_coles_with_xgboost()
        print("\nDixon-Coles (XGBoost) trained and saved successfully")
    except Exception as e:
        print(f"\nError training Dixon-Coles (XGBoost): {e}")
    
    # Step 3: Train Dixon-Coles with CatBoost
    print("\n\n[3/4] Training Dixon-Coles with CatBoost Base")
    print("-"*70)
    try:
        dc_cat = train_dixon_coles_with_catboost()
        print("\nDixon-Coles (CatBoost) trained and saved successfully")
    except Exception as e:
        print(f"\nError training Dixon-Coles (CatBoost): {e}")
    
    # Step 4: Comprehensive Evaluation
    print("\n\n[4/4] Running Comprehensive Evaluation")
    print("-"*70)
    try:
        evaluator = ModelEvaluator(
            models_dir='./models',
            data_path='./data/matches_with_features.csv'
        )
        results = evaluator.run_full_evaluation()
        print("\nEvaluation complete successfully")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
    
    # Summary
    print("\n\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\nModels saved in ./models/:")
    print("  - poisson_model.pkl           (Poisson Regression)")
    print("  - dixon_coles_xgb.pkl         (Dixon-Coles for XGBoost)")
    print("  - dixon_coles_cat.pkl         (Dixon-Coles for CatBoost)")
    print("\nResults saved in ./models/results/:")
    print("  - model_comparison.csv        (Performance comparison)")
    print("\nNext steps:")
    print("  1. Use src/prediction/predict.py for match predictions")
    print("  2. Implement tournament simulation (Phase 4)")
    print("="*70)


if __name__ == "__main__":
    main()
