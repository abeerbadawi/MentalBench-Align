#!/usr/bin/env python3
"""
Alignment Factor Analysis for MentalBench-10

This script calculates the Alignment Factor (AF) between human and LLM judge evaluations
as described in the MentalBench-10 paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

class AlignmentFactorAnalyzer:
    """
    Analyzes alignment between human and LLM judge evaluations.
    """
    
    def __init__(self, human_data_path: str, llm_data_path: str):
        """
        Initialize the analyzer with human and LLM evaluation data.
        
        Args:
            human_data_path: Path to human evaluation results
            llm_data_path: Path to LLM judge evaluation results
        """
        self.human_data = pd.read_csv(human_data_path)
        self.llm_data = pd.read_csv(llm_data_path)
        self.attributes = [
            'Guidance', 'Informativeness', 'Relevance', 'Safety',
            'Empathy', 'Helpfulness', 'Understanding'
        ]
    
    def calculate_alignment_factor(self, llm_judge: str) -> Dict[str, float]:
        """
        Calculate Alignment Factor for a specific LLM judge.
        
        Args:
            llm_judge: Name of the LLM judge
            
        Returns:
            Dictionary with AF scores for each attribute
        """
        # Filter data for the specific judge
        judge_data = self.llm_data[self.llm_data['judge'] == llm_judge]
        
        af_scores = {}
        for attr in self.attributes:
            # Calculate absolute differences
            differences = np.abs(
                judge_data[f'{attr}_llm'] - judge_data[f'{attr}_human']
            )
            # Calculate AF as mean absolute error
            af_scores[attr] = differences.mean()
        
        return af_scores
    
    def compare_judges(self, judges: List[str]) -> pd.DataFrame:
        """
        Compare alignment factors across multiple judges.
        
        Args:
            judges: List of judge names to compare
            
        Returns:
            DataFrame with AF scores for each judge and attribute
        """
        results = []
        for judge in judges:
            af_scores = self.calculate_alignment_factor(judge)
            af_scores['Judge'] = judge
            results.append(af_scores)
        
        return pd.DataFrame(results)
    
    def plot_alignment_heatmap(self, judges: List[str], save_path: str = None):
        """
        Create a heatmap visualization of alignment factors.
        
        Args:
            judges: List of judge names
            save_path: Optional path to save the plot
        """
        df = self.compare_judges(judges)
        df_plot = df.set_index('Judge')[self.attributes]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_plot, annot=True, cmap='RdYlBu_r', center=0.5)
        plt.title('Alignment Factor Heatmap: Human vs LLM Judges')
        plt.xlabel('Evaluation Attributes')
        plt.ylabel('LLM Judges')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, judges: List[str]) -> Dict:
        """
        Generate a comprehensive summary report.
        
        Args:
            judges: List of judge names
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.compare_judges(judges)
        
        summary = {
            'best_aligned_judge': df[self.attributes].mean(axis=1).idxmin(),
            'worst_aligned_judge': df[self.attributes].mean(axis=1).idxmax(),
            'easiest_attribute': df[self.attributes].mean().idxmin(),
            'hardest_attribute': df[self.attributes].mean().idxmax(),
            'overall_alignment': df[self.attributes].mean().mean(),
            'judge_rankings': df[self.attributes].mean(axis=1).sort_values().to_dict()
        }
        
        return summary

def main():
    """
    Main function to demonstrate alignment analysis.
    """
    # Example usage
    analyzer = AlignmentFactorAnalyzer(
        human_data_path='results/human_evaluation/human_scores.csv',
        llm_data_path='results/llm_judge_results/llm_scores.csv'
    )
    
    # Define judges to compare
    judges = ['GPT-4o', 'GPT-4o-Mini', 'Claude-3.7-Sonnet', 'Gemini-2.5-Flash']
    
    # Generate analysis
    summary = analyzer.generate_summary_report(judges)
    
    # Print results
    print("=== MentalBench-10 Alignment Factor Analysis ===")
    print(f"Best aligned judge: {summary['best_aligned_judge']}")
    print(f"Worst aligned judge: {summary['worst_aligned_judge']}")
    print(f"Easiest attribute to align: {summary['easiest_attribute']}")
    print(f"Hardest attribute to align: {summary['hardest_attribute']}")
    print(f"Overall alignment score: {summary['overall_alignment']:.3f}")
    
    # Create visualization
    analyzer.plot_alignment_heatmap(judges, 'results/alignment_analysis/alignment_heatmap.png')
    
    # Save detailed results
    with open('results/alignment_analysis/alignment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
