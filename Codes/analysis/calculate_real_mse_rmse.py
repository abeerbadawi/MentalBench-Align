#!/usr/bin/env python3
"""
Calculate Real MSE and RMSE from Individual Rating Pairs
Based on the existing ICC analysis code structure.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Constants from your main code
ATTRIBUTES = [
    'Guidance', 'Informativeness', 'Relevance', 'Safety',
    'Empathy', 'Helpfulness', 'Understanding'
]

MODEL_NAME_TO_KEY = {
    'Claude-3.5-Haiku': 'claude',
    'deepseek-llama': 'ds_llama',
    'deepseek-qwen': 'ds_qwen',
    'Gemini 2.0-Flash': 'gemini',
    'gpt-4o': 'gpt4o',
    'gpt-4omini': 'gpt4omini',
    'Llama-3.1': 'llama_3',
    'Qwen-2.5': 'qwen_2',
    'Qwen-3': 'qwen_3',
    'Human': 'human',
}

JUDGE_TO_FILE = {
    'Claude': 'claude_llm_judge_with_human.csv',
    'GPT-4o': 'gpt-4o_llm_judge_with_human.csv',
    'Gemini': 'gemini_llm_judge_with_human.csv',
    'o4-mini': 'o4-mini_llm_judge_with_human.csv',
}

JUDGE_EXCLUSIONS = {
    'Claude': ['Claude-3.5-Haiku'],
    'GPT-4o': ['gpt-4o'],
    'Gemini': ['Gemini 2.0-Flash'],
    'o4-mini': ['gpt-4omini'],
}


def _parse_rating(json_str: str, attribute: str) -> float:
    """Parse JSON rating for a specific attribute."""
    try:
        if pd.isna(json_str) or json_str == '':
            return np.nan
        s = str(json_str).strip()
        if s.startswith('```json'):
            s = s.replace('```json', '').replace('```', '').strip()
        data = json.loads(s)
        val = data.get(attribute, np.nan)
        if val is None or val == '':
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def _collect_rating_pairs(df: pd.DataFrame, model_keys: List[str], attribute: str) -> Tuple[List[float], List[float]]:
    """
    Collect all individual rating pairs for MSE/RMSE calculation.
    This is the key function from your code that gets individual ratings.
    """
    human_ratings = []
    llm_ratings = []
    
    for model_key in model_keys:
        hcol = f'human_response_{model_key}'
        jcol = f'judge_response_{model_key}'
        
        if hcol not in df.columns or jcol not in df.columns:
            continue
            
        for _, row in df.iterrows():
            h = _parse_rating(row[hcol], attribute)
            l = _parse_rating(row[jcol], attribute)
            if not (np.isnan(h) or np.isnan(l)):
                human_ratings.append(h)
                llm_ratings.append(l)
    
    return human_ratings, llm_ratings


def calculate_real_mse_rmse():
    """Calculate real MSE and RMSE from individual rating pairs."""
    
    results = []
    
    for judge, csv_path in JUDGE_TO_FILE.items():
        try:
            df = pd.read_csv(csv_path)
            print(f"Processing {judge}...")
        except FileNotFoundError:
            print(f"Warning: Missing file {csv_path}")
            continue

        # Get exclusion list for this judge
        excluded_models = JUDGE_EXCLUSIONS.get(judge, [])
        
        for attr in ATTRIBUTES:
            print(f"  Calculating MSE/RMSE for {attr}...")
            
            # Get valid model keys (excluding self-evaluations)
            valid_model_keys = []
            for model_name, model_key in MODEL_NAME_TO_KEY.items():
                if model_name not in excluded_models:
                    valid_model_keys.append(model_key)
            
            # Collect individual rating pairs
            human_ratings, llm_ratings = _collect_rating_pairs(df, valid_model_keys, attr)
            
            if len(human_ratings) < 2:
                # Not enough data
                results.append({
                    'judge': judge,
                    'attribute': attr,
                    'n_rating_pairs': 0,
                    'mse': np.nan,
                    'rmse': np.nan,
                    'bias': np.nan,
                    'human_mean': np.nan,
                    'llm_mean': np.nan,
                    'human_std': np.nan,
                    'llm_std': np.nan
                })
                continue
            
            # Convert to numpy arrays
            human_array = np.array(human_ratings)
            llm_array = np.array(llm_ratings)
            
            # Calculate MSE and RMSE
            differences = llm_array - human_array
            mse = np.mean(differences ** 2)
            rmse = np.sqrt(mse)
            
            # Additional statistics
            bias = np.mean(differences)  # Mean bias (LLM - Human)
            human_mean = np.mean(human_array)
            llm_mean = np.mean(llm_array)
            human_std = np.std(human_array, ddof=1)
            llm_std = np.std(llm_array, ddof=1)
            
            results.append({
                'judge': judge,
                'attribute': attr,
                'n_rating_pairs': len(human_ratings),
                'mse': mse,
                'rmse': rmse,
                'bias': bias,
                'human_mean': human_mean,
                'llm_mean': llm_mean,
                'human_std': human_std,
                'llm_std': llm_std
            })
    
    return pd.DataFrame(results)


def main():
    """Main function to calculate and save MSE/RMSE results."""
    
    print("="*60)
    print("CALCULATING REAL MSE AND RMSE FROM INDIVIDUAL RATING PAIRS")
    print("="*60)
    
    # Calculate MSE/RMSE
    mse_rmse_df = calculate_real_mse_rmse()
    
    # Save results
    output_file = Path('real_mse_rmse_results.csv')
    mse_rmse_df.to_csv(output_file, index=False)
    
    print(f"\n Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    valid_results = mse_rmse_df[~mse_rmse_df['mse'].isna()]
    
    if len(valid_results) > 0:
        print(f"\nValid judge-attribute pairs: {len(valid_results)}")
        print(f"Total rating pairs analyzed: {valid_results['n_rating_pairs'].sum()}")
        
        print(f"\nMSE Statistics:")
        print(f"  Mean: {valid_results['mse'].mean():.4f}")
        print(f"  Std:  {valid_results['mse'].std():.4f}")
        print(f"  Min:  {valid_results['mse'].min():.4f}")
        print(f"  Max:  {valid_results['mse'].max():.4f}")
        
        print(f"\nRMSE Statistics:")
        print(f"  Mean: {valid_results['rmse'].mean():.4f}")
        print(f"  Std:  {valid_results['rmse'].std():.4f}")
        print(f"  Min:  {valid_results['rmse'].min():.4f}")
        print(f"  Max:  {valid_results['rmse'].max():.4f}")
        
        print(f"\nBias Statistics:")
        print(f"  Mean: {valid_results['bias'].mean():.4f}")
        print(f"  Std:  {valid_results['bias'].std():.4f}")
        print(f"  Min:  {valid_results['bias'].min():.4f}")
        print(f"  Max:  {valid_results['bias'].max():.4f}")
        
        # Show top and bottom cases
        print(f"\nHighest MSE:")
        worst_mse = valid_results.loc[valid_results['mse'].idxmax()]
        print(f"  {worst_mse['judge']} - {worst_mse['attribute']}: MSE={worst_mse['mse']:.4f}, RMSE={worst_mse['rmse']:.4f}")
        
        print(f"\nLowest MSE:")
        best_mse = valid_results.loc[valid_results['mse'].idxmin()]
        print(f"  {best_mse['judge']} - {best_mse['attribute']}: MSE={best_mse['mse']:.4f}, RMSE={best_mse['rmse']:.4f}")
        
    else:
        print("No valid results found!")
    
    print("\n" + "="*60)
    print("REAL MSE/RMSE CALCULATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
