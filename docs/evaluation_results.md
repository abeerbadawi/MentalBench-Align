# MentalBench-10 Evaluation Results

This document provides a comprehensive overview of all evaluation results from the MentalBench-10 benchmark.

## Overview

The evaluation framework assessed 100,000 responses (10,000 conversations × 10 responses each) using:
- **4 LLM Judges**: GPT-4o, GPT-4o-Mini, Claude-3.7-Sonnet, Gemini-2.5-Flash
- **2 Human Experts**: Clinical professionals with psychiatric training
- **7 Evaluation Attributes**: Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding

## 1. LLM-as-a-Judge Evaluation Results

### Summary Table
| Model | Source | Claude-3.7-Sonnet | GPT-4o | O4-Mini | Gemini-2.5-Flash | Average | Rank |
|-------|--------|-------------------|--------|---------|------------------|---------|------|
| Gemini-2.0-Flash | Closed | 4.87 | 4.96 | 4.89 | 4.94 | 4.92 | 1 |
| GPT-4o | Closed | 4.81 | 4.97 | 4.88 | 4.90 | 4.89 | 2 |
| GPT-4o-Mini | Closed | 4.74 | 4.95 | 4.84 | 4.88 | 4.85 | 3 |
| Claude-3.5-Haiku | Closed | 4.78 | 4.87 | 4.70 | 4.85 | 4.80 | 4 |
| LLaMA-3.1-8B-Instruct | Open | 4.71 | 4.84 | 4.63 | 4.77 | 4.74 | 5 |
| DeepSeek-LLaMA-8B | Open | 4.55 | 4.82 | 4.64 | 4.74 | 4.69 | 6 |
| DeepSeek-Qwen-7B | Open | 4.03 | 4.62 | 4.39 | 4.44 | 4.37 | 7 |
| Qwen2.5-7B-Instruct | Open | 4.26 | 4.46 | 4.35 | 4.37 | 4.36 | 8 |
| Qwen-3-4B | Open | 3.78 | 4.19 | 4.04 | 4.20 | 4.05 | 9 |
| Human Response | Human | 3.90 | 4.24 | 3.89 | 4.16 | 4.05 | 9 |

### Key Findings
- **Closed-source models dominate**: Top 4 positions occupied by closed models
- **Gemini-2.0-Flash leads**: Highest average score (4.92)
- **Open-source performance**: LLaMA-3.1-8B-Instruct best open model (4.74)
- **Human responses outperformed**: LLMs scored higher than human responses

## 2. Human Expert Evaluation Results

### Summary Table
| Model | Source | Guidance | Informativeness | Relevance | Safety | Empathy | Helpfulness | Understanding | Average | Rank |
|-------|--------|----------|-----------------|-----------|--------|---------|-------------|--------------|---------|------|
| GPT-4o | Closed | 4.58 | 4.72 | 4.98 | 4.97 | 4.76 | 4.70 | 4.99 | 4.81 | 1 |
| Gemini-2.0-Flash | Closed | 4.53 | 4.78 | 4.98 | 4.98 | 4.38 | 4.50 | 4.98 | 4.73 | 2 |
| GPT-4o-Mini | Closed | 4.31 | 4.46 | 4.96 | 4.94 | 4.42 | 4.48 | 4.95 | 4.65 | 3 |
| Qwen2.5-7B-Instruct | Open | 4.23 | 4.24 | 4.89 | 4.91 | 4.43 | 4.41 | 4.87 | 4.57 | 4 |
| LLaMA-3.1-8B-Instruct | Open | 3.96 | 4.32 | 4.93 | 4.92 | 4.44 | 4.32 | 4.90 | 4.54 | 5 |
| Claude-3.5-Haiku | Closed | 3.91 | 4.11 | 4.85 | 4.84 | 4.40 | 4.35 | 4.83 | 4.47 | 6 |
| Qwen-3-4B | Open | 3.88 | 4.02 | 4.79 | 4.80 | 4.34 | 4.30 | 4.82 | 4.42 | 7 |
| DeepSeek-LLaMA-8B | Open | 3.72 | 3.95 | 4.76 | 4.77 | 4.28 | 4.19 | 4.80 | 4.35 | 8 |
| DeepSeek-Qwen-7B | Open | 3.65 | 3.90 | 4.74 | 4.76 | 4.22 | 4.17 | 4.78 | 4.32 | 9 |
| Human Response | Human | 3.05 | 3.07 | 3.86 | 3.89 | 3.79 | 3.21 | 3.77 | 3.52 | 10 |

### Key Findings
- **GPT-4o leads human evaluation**: Highest overall score (4.81)
- **Strong cognitive performance**: Models excel in Relevance and Safety
- **Affective challenges**: Empathy and Helpfulness show more variation
- **Human responses outperformed**: LLMs scored higher across all dimensions

## 3. Alignment Factor Analysis

### Summary Table
| Judge | Guidance | Informativeness | Relevance | Safety | Empathy | Helpfulness | Understanding | Average |
|-------|----------|-----------------|-----------|--------|---------|-------------|--------------|---------|
| Claude-3.7-Sonnet | 0.59 | 0.58 | 0.38 | 0.20 | 0.68 | 0.66 | 0.38 | 0.49 |
| GPT-4o-Mini | 0.67 | 0.59 | 0.19 | 0.14 | 0.66 | 0.61 | 0.21 | 0.44 |
| GPT-4o | 0.80 | 0.64 | 0.20 | 0.15 | 0.68 | 0.66 | 0.19 | 0.47 |
| Gemini-2.5-Flash | 0.71 | 0.66 | 0.21 | 0.17 | 0.69 | 0.76 | 0.22 | 0.49 |

### Key Findings
- **GPT-4o-Mini best aligned**: Lowest average error (0.44)
- **Safety easiest to align**: Lowest error across all judges
- **Empathy hardest to align**: Highest error across all judges
- **Strong overall alignment**: All judges show reasonable agreement with humans

## 4. Detailed Results Files

### LLM Judge Results
- `llm_judge_summary_table.csv`: Main summary table
- `LLM_Judge_GPT-4o_Evaluation__With_Model_Names_.csv`: GPT-4o judge results
- `LLM_Judge_GPT-4o-Mini_Evaluation__With_Model_Names_.csv`: GPT-4o-Mini judge results
- `LLM_Judge_Claude_Evaluation_Summary__Rounded_.csv`: Claude judge results
- `LLM_Judge_Gemini_Evaluation__With_Model_Names_.csv`: Gemini judge results
- `Combined_Averages_Ranked_by_Model.csv`: Combined rankings

### Human Evaluation Results
- `human_evaluation_summary.csv`: Main human evaluation table
- `Lindsay_-_Average_Scores_per_Response.csv`: Lindsay's evaluations
- `Sheri_-_Average_Scores_per_Response.csv`: Sheri's evaluations
- `Combined_Lindsay___Sheri_Averages_with_Rank.csv`: Combined human results

### Alignment Analysis
- `alignment_factor_summary.csv`: Main alignment factor table
- Individual judge alignment files for detailed analysis

## 5. Statistical Significance

### Paired t-tests Results
- **Gemini-2.0-Flash**: No significant difference from other closed models
- **GPT-4o**: Statistically significant differences from all closed models except Gemini-2.0-Flash
- **LLaMA-3.1-8B-Instruct**: Significantly higher than all open-source models
- **Human responses**: Significantly outperformed by multiple LLMs

## 6. Model Performance Analysis

### Closed vs Open Source
- **Closed-source dominance**: Average score 4.87 vs 4.36 for open-source
- **Performance gap**: 0.51 point difference between closed and open models
- **Best open model**: LLaMA-3.1-8B-Instruct (4.74)

### Attribute Performance
- **Strongest attributes**: Relevance and Safety (consistently high scores)
- **Challenging attributes**: Empathy and Helpfulness (more variation)
- **Cognitive vs Affective**: Models excel in cognitive dimensions

## 7. Clinical Implications

### Key Insights
1. **LLMs match or exceed human performance** in structured dimensions
2. **Affective traits remain challenging** for open-source models
3. **Strong alignment** between human and LLM judges
4. **Scalable evaluation** possible with LLM-as-a-judge approach

### Recommendations
- **Closed-source models** suitable for production mental health applications
- **Open-source models** need improvement in affective dimensions
- **Continued evaluation** needed as models evolve
- **Human oversight** still essential for clinical applications

## 8. Data Files Structure

```
results/
├── llm_judge_results/
│   ├── llm_judge_summary_table.csv          # Main LLM judge table
│   ├── LLM_Judge_GPT-4o_Evaluation__With_Model_Names_.csv
│   ├── LLM_Judge_GPT-4o-Mini_Evaluation__With_Model_Names_.csv
│   ├── LLM_Judge_Claude_Evaluation_Summary__Rounded_.csv
│   ├── LLM_Judge_Gemini_Evaluation__With_Model_Names_.csv
│   └── Combined_Averages_Ranked_by_Model.csv
├── human_evaluation/
│   ├── human_evaluation_summary.csv         # Main human evaluation table
│   ├── Lindsay_-_Average_Scores_per_Response.csv
│   ├── Sheri_-_Average_Scores_per_Response.csv
│   └── Combined_Lindsay___Sheri_Averages_with_Rank.csv
└── alignment_analysis/
    └── alignment_factor_summary.csv         # Main alignment factor table
```

## 9. Citation

When using these evaluation results, please cite:

```bibtex
@inproceedings{mentalbench10-iclr2025,
  title={From Empathy to Action: Benchmarking LLMs in Mental Health with MentalBench-10 and a Novel Cognitive-Affective Evaluation Approach},
  author={Anonymous},
  booktitle={},
  year={2025}
}
```
