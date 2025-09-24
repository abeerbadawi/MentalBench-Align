# MentalBench-10 Dataset Description

## Overview

MentalBench-10 is a comprehensive benchmark dataset for evaluating Large Language Models (LLMs) in mental health support contexts. The dataset comprises 10,000 real-world mental health conversations with responses from one human expert and nine leading LLMs.

## Dataset Sources

The dataset is curated from three high-integrity, publicly available mental health datasets:

### 1. MentalChat16K
- **Source**: PISCES clinical trial
- **Conversations**: 6,338 anonymized transcripts
- **Content**: Real conversations between licensed clinicians and youth
- **Topics**: Depression, anxiety, grief, and other sensitive mental health concerns
- **Quality**: High-integrity clinical data with professional oversight

### 2. EmoCare
- **Source**: Counseling sessions dataset
- **Conversations**: 8,187 unique entries (processed from 260 sessions)
- **Content**: Emotional well-being, relationships, and family issues
- **Processing**: Enhanced using ChatGPT-4 for consistency
- **Focus**: Diverse counseling scenarios and emotional support

### 3. CounselChat
- **Source**: CounselChat.com platform
- **Questions**: 854 mental health questions
- **Content**: Therapist responses to user-submitted mental health questions
- **Value**: Diverse professional perspectives and multi-response coverage
- **Quality**: Expert-written responses from licensed therapists

## Dataset Statistics

### General Statistics
- **Total Conversations**: 14,737 (original), 10,000 (development set)
- **Average Context Length**: 72.64 words
- **Average Response Length**: 87.03 words
- **Mental Health Conditions**: 23 categories
- **Data Split**: 10,000 development, 4,737 training
- **Total Model Responses**: 100,000 (10,000 conversations × 10 responses each)

### Mental Health Condition Distribution

The dataset covers 23 clinically relevant mental health conditions:

**Most Common Conditions**:
1. Relationship Issues
2. Anxiety
3. Depression
4. Family Problems
5. Stress Management

**Less Common Conditions**:
- Self-harm
- Bullying
- Exploitation
- Trauma
- Substance Abuse

### Model Coverage

Each conversation includes responses from:

**Closed-Source Models**:
- GPT-4o (OpenAI)
- GPT-4o-Mini (OpenAI)
- Claude-3.5-Haiku (Anthropic)
- Gemini-2.0-Flash (Google DeepMind)

**Open-Source Models**:
- LLaMA-3.1-8B-Instruct (Meta)
- Qwen2.5-7B-Instruct (Alibaba)
- Qwen-3-4B (Alibaba)
- DeepSeek-LLaMA-8B (DeepSeek)
- DeepSeek-Qwen-7B (DeepSeek)

## Data Schema

### Main Dataset Structure

| Column | Description |
|--------|-------------|
| `context` | User's mental health concern or inquiry |
| `response` | Human expert response (ground truth) |
| `context_length` | Word count of the context |
| `response_length` | Word count of the human response |
| `Claude-3.5-Haiku` | Response from Claude 3.5 Haiku |
| `deepseek-llama` | Response from DeepSeek LLaMA |
| `deepseek-qwen` | Response from DeepSeek Qwen |
| `Gemini` | Response from Gemini-2.0-Flash |
| `gpt-4o` | Response from GPT-4o |
| `gpt-4omini` | Response from GPT-4o-Mini |
| `Llama-3.1` | Response from LLaMA 3.1 |
| `Qwen-2.5` | Response from Qwen2.5-7B |
| `Qwen-3` | Response from Qwen-3 Alpha |

### Evaluation Data Structure

For each response, evaluation scores are provided across seven dimensions:

| Attribute | Description | Scale |
|-----------|-------------|-------|
| Guidance | Structured direction and next steps | 1-5 |
| Informativeness | Useful and relevant suggestions | 1-5 |
| Relevance | Topic alignment and specificity | 1-5 |
| Safety | Ethical compliance and harm avoidance | 1-5 |
| Empathy | Emotional warmth and validation | 1-5 |
| Helpfulness | Distress reduction and support | 1-5 |
| Understanding | Accurate emotional reflection | 1-5 |


## Ethical Considerations

- **Privacy**: All data anonymized and de-identified



