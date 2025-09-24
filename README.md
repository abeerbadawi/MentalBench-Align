# MentalBench-100k & MentalAlign-70k: Reliable Evaluation of LLMs in Mental Health Dialogue  

This repository contains the datasets, code, and evaluation pipeline for the paper **"WHEN CAN WE TRUST LLMS IN MENTAL HEALTH? LARGE-SCALE BENCHMARKS FOR RELIABLE DIALOGUE EVALUATION AND ALIGNMENT"**.  

<img width="2870" height="1588" alt="framework" src="https://github.com/user-attachments/assets/17fe70cb-c890-4bf3-9905-1b5fb7f94f59" />

---

## Overview  

We introduce two complementary benchmarks for systematically evaluating LLMs in mental health support:  

- **MentalBench-100k**: 10,000 authentic therapeutic conversations paired with 9 diverse LLM responses each (100,000 responses total).  
- **MentalAlign-70k**: 70,000 ratings across 7 attributes from both human clinical experts and LLM judges, enabling the first large-scale human–AI comparison of evaluation reliability.  

Together, these resources establish a dual-benchmark ecosystem for studying **response generation** and **evaluation alignment** in mental health contexts.  

---

## Key Contributions  

1. **MentalBench-100k Dataset**: Largest benchmark of authentic single-session therapeutic dialogues with 9 LLM-generated responses per conversation.  
2. **MentalAlign-70k Dataset**: Dual-axis evaluation (Cognitive Support Score, Affective Resonance Score) with 70,000 ratings from experts and 4 LLM judges.  
3. **Affective–Cognitive Alignment Framework**: Reliability-oriented methodology using Intraclass Correlation Coefficients (ICC), bootstrap confidence intervals, and bias analysis to quantify agreement magnitude and precision.  
4. **Reliability Guidance**: First evidence-based recommendations for when LLM-as-a-judge evaluation can be trusted and when human oversight is essential.  

---

## Repository Structure

```
├── MentalBench-100k/              # Dataset files
│   ├── MentalBench-100k.csv       # Main dataset with conversations and responses
│   └── metadata/                 # Dataset metadata and documentation
├── MentalAlign-70k/              # Human and LLMs as a judge evalaution results
│   ├──LLM as a judge Evaluation/     # LLMs as a judge Evaluation Results
│      ├── claude-3-7-sonnet_llm_judge.csv   # Claude as LLM as a judge
│      ├── gemini-2.5-flash_llm_judge.csv    # Gemini as LLM as a judge
│      ├── gpt-4o-llm_judge.csv      # GPT 4o as LLM as a judge
│      └── o4-mini-llm_judge.csv     # o4-mini as LLM as a judge
    ├── Human Evaluation /         # Human Evaluation Results
│       ├── HumanEvaluation.csv   # Humans as a judge
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
├── 
└── README.md                      # This file
```



## Dataset  

### MentalBench-100k  

- **Conversations**: 10,000  
- **Responses per conversation**: 1 human + 9 LLMs  
- **Total responses**: 100,000  
- **Conditions covered**: 23 clinically relevant categories (anxiety, depression, relationships, grief, etc.)  
- **Average context length**: 72.6 words  
- **Average response length**: 87.0 words  

**LLMs used**: GPT-4o, GPT-4o-Mini, Claude-3.5-Haiku, Gemini-2.0-Flash, LLaMA-3.1-8B-Instruct, Qwen2.5-7B, Qwen-3-4B, DeepSeek-LLaMA-8B, DeepSeek-Qwen-7B.  

---

### MentalAlign-70k  

- **Ratings**: 70,000 across 1,000 conversations × 10 responses × 7 attributes  
- **Judges**: 3 human experts + 4 LLMs (Claude-3.7-Sonnet, GPT-4o, GPT-4o-Mini, Gemini-2.5-Flash)  
- **Attributes**:  
  - **Cognitive Support Score (CSS)**: Guidance, Informativeness, Relevance, Safety  
  - **Affective Resonance Score (ARS)**: Empathy, Helpfulness, Understanding  

---

## Evaluation Framework  

- **ICC Analysis**: Agreement and consistency between human and LLM judges.  
- **Bootstrap Confidence Intervals**: Quantify precision of reliability estimates.  
- **Bias Detection**: Attribute- and model-specific inflation analysis.  
- **Reliability Categories**:  
  - Good Reliability (GR)  
  - Moderate Validation Needed (MV)  
  - Limited Reliability (LR)  

This framework reveals where automated evaluation is reliable (e.g., Guidance, Informativeness) and where human oversight is mandatory (e.g., Empathy, Safety, Relevance).  

---

## Results (Highlights)  

- **High-capacity models** (GPT-4o, Gemini-2.0-Flash) consistently outperform smaller open-source systems.  
- **Empathy & Helpfulness** show deceptively high scores but wide uncertainty, requiring caution.  
- **Safety & Relevance** exhibit systematically poor reliability across judges.  
- **LLM Judges** systematically inflate ratings (+0.4–0.8 on affective attributes).  

---

## Usage  

### Setup  

```bash
# Clone the repository
git clone https://github.com/your-username/mentalbench-align.git
cd mentalbench-align

# Install dependencies
pip install -r requirements.txt

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
@inproceedings{mentalbench-iclr2025,
  title={When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable Dialogue Evaluation and Alignment},
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

