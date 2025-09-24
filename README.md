# MentalBench-10: Benchmarking LLMs in Mental Health Support

This repository contains the code, data, and evaluation results for the paper **"From Empathy to Action: Benchmarking LLMs in Mental Health with MentalBench-10 and a Novel Cognitive-Affective Evaluation Approach"**.
<img width="2870" height="1588" alt="framework2" src="https://github.com/user-attachments/assets/17fe70cb-c890-4bf3-9905-1b5fb7f94f59" />

## Overview

MentalBench-10 is a large-scale real-world benchmark for evaluating Large Language Models (LLMs) in mental health support contexts. The benchmark comprises 10,000 real-world mental health conversations with responses from one human expert and nine leading LLMs.

## Key Contributions

1. **MentalBench-10 Dataset**: 10,000 real-world mental health conversations with human and LLM responses
2. **Novel Evaluation Framework**: Dual-axis evaluation using Cognitive Support Score (CSS) and Affective Resonance Score (ARS)
3. **LLM-as-a-Judge Evaluation**: Scalable evaluation using four high-performing LLM judges
4. **Alignment Factor**: Metric to measure agreement between human and LLM-based ratings

## Repository Structure

```
├── data/                          # Dataset files
│   ├── MentalBench-10.csv        # Main dataset with conversations and responses
│   └── metadata/                  # Dataset metadata and documentation
├── code/                          # Implementation code
│   ├── generation/                # LLM response generation scripts
│   ├── evaluation/                # Evaluation framework implementation
│   └── analysis/                  # Analysis and visualization scripts
├── results/                       # Evaluation results
│   ├── llm_judge_results/        # LLM-as-a-judge evaluation results
│   ├── human_evaluation/         # Human expert evaluation results
│   └── alignment_analysis/       # Alignment factor analysis
├── docs/                          # Documentation
│   ├── evaluation_guidelines.md   # Evaluation criteria and guidelines
│   └── dataset_description.md    # Dataset description and statistics
├── LLM as a judge Evaluation/     # LLMs as a judge Evaluation Results
│   ├── claude-3-7-sonnet_llm_judge.csv   # Claude as LLM as a judge
│   ├── gemini-2.5-flash_llm_judge.csv    # Gemini as LLM as a judge
│   ├── gpt-4o-llm_judge.csv      # GPT 4o as LLM as a judge
│   └── o4-mini-llm_judge.csv     # o4-mini as LLM as a judge
├── Human Evaluation /     # Human Evaluation Results
│   ├── HumanEvaluation.csv   # Humans as a judge
└── README.md                      # This file
```

## Dataset

### MentalBench-10 Dataset

The dataset contains 10,000 real-world mental health conversations sourced from three high-integrity datasets:
- **MentalChat16K**: 6,338 conversations from PISCES clinical trial
- **EmoCare**: 8,187 counseling sessions
- **CounselChat**: 854 therapist responses

  We cleaned the dataset, and from the 15379 conversations we used 10000 for evaluation. 

Each conversation includes:
- User context (mental health concern)
- Human expert response
- Responses from 9 LLMs that we generated (GPT-4o, GPT-4o-Mini, Claude-3.5-Haiku, Gemini-2.0-Flash, LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen-3-4B, DeepSeek-LLaMA-8B, DeepSeek-Qwen-7B)

### Dataset Statistics

- **Total Conversations**: 10,000
- **Average Context Length**: 72.64 words
- **Average Response Length**: 87.03 words
- **Mental Health Conditions**: 23 categories (anxiety, depression, relationship issues, etc.)

## Evaluation Framework

### Dual-Axis Evaluation

1. **Cognitive Support Score (CSS)**:
   - Guidance: Structured direction and next steps
   - Informativeness: Useful and relevant suggestions
   - Relevance: Topic alignment and specificity
   - Safety: Ethical compliance and harm avoidance

2. **Affective Resonance Score (ARS)**:
   - Empathy: Emotional warmth and validation
   - Helpfulness: Distress reduction and support
   - Understanding: Accurate emotional reflection

### Evaluation Methods

1. **LLM-as-a-Judge**: Four LLM judges (GPT-4o, GPT-4o-Mini, Claude-3.7-Sonnet, Gemini-2.5-Flash) evaluated 10,000 conversations with a total of 100,000 responses
2. **Human Expert Evaluation**: Two clinical experts evaluated 2,500 responses (250 conversations)
3. **Alignment Factor**: Measures agreement between human and LLM-based ratings

## Results

### Key Findings

1. **Closed-source models dominate**: Gemini-2.0-Flash (4.92), GPT-4o (4.89), GPT-4o-Mini (4.85)
2. **Open-source performance**: LLaMA-3.1-8B-Instruct (4.74) leads open-source models
3. **Human responses outperformed**: LLMs scored higher than human responses across most dimensions
4. **Affective challenges**: Empathy and helpfulness remain challenging for open-source models

### Model Rankings

| Rank | Model | Type | Average Score |
|------|-------|------|---------------|
| 1 | Gemini-2.0-Flash | Closed | 4.92 |
| 2 | GPT-4o | Closed | 4.89 |
| 3 | GPT-4o-Mini | Closed | 4.85 |
| 4 | Claude-3.5-Haiku | Closed | 4.80 |
| 5 | LLaMA-3.1-8B-Instruct | Open | 4.74 |

## Usage

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/mentalbench-10.git
cd mentalbench-10

# Install dependencies
pip install -r requirements.txt
```

### Running Evaluations

```python
# Generate LLM responses
python code/generation/generate_responses.py

# Run LLM-as-a-judge evaluation
python code/evaluation/llm_judge_evaluation.py

# Calculate alignment factors
python code/analysis/alignment_analysis.py
```

## Citation

```bibtex
@inproceedings{mentalbench10-iclr2025,
  title={From Empathy to Action: Benchmarking LLMs in Mental Health with MentalBench-10 and a Novel Cognitive-Affective Evaluation Approach},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Ethical Considerations

This work involves sensitive mental health data and AI-generated responses. Key considerations:

- All datasets were publicly available and anonymized
- Models are not intended to replace human therapists
- Potential for demographic and cultural biases
- Emotional burden on human annotators acknowledged
- Focus on responsible AI deployment in mental health

## Limitations

- Computational cost constraints limited model exploration
- Human evaluation on 250 conversations (could be expanded)
- Quality of human baseline responses from existing datasets
- Potential bias in LLM-as-a-judge evaluation
- Model performance may vary with different prompts
