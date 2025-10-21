### 📍 RQ3: Systematic Bias Decomposition — Where Do LLM Judges Drift from Human Standards?

While RQ2 focused on *reliability* (i.e., consistency and precision), RQ3 investigates **systematic bias** — persistent deviations between LLM ratings and human expert scores.

We distinguish between:
| Error Type | What It Means | Can It Be Fixed? |
|-----------|--------------|------------------|
| ✅ **Systematic Bias** | Consistent over/under-scoring patterns | Potentially correctable via calibration |
| ❌ **Random Error** | Unstable, unpredictable variation | Not easily fixable — indicates fundamental unreliability |

---

### ✅ How We Analyzed Bias

For each attribute, we computed:
| Metric | Description |
|--------|------------|
| **Human Mean** | Average human rating (baseline) |
| **LLM Mean** | Average rating by the judge |
| **Bias** | LLM − Human (positive = inflation) |
| **MSE** | Error magnitude vs human truth |

📌 We examined 4 judges × 7 attributes = 28 judge–attribute bias cases.

---

### 🎯 Key Findings

| Dimension | Trend | Interpretation |
|-----------|--------|----------------|
| 🧠 **Cognitive attributes** (Guidance, Informativeness) | Small/moderate bias (+0.25 to +0.46) | ✅ Suitable for **calibration-based correction** |
| 💛 **Affective attributes** (Empathy, Helpfulness) | Inflated bias (+0.40 to +0.80) | ⚠️ Risk of exaggerated empathy → “false warmth” |
| ⚠️ **Safety & Relevance** | Modest bias (+0.18 to +0.39) but unreliable | ❌ Bias correction not enough due to poor reliability (from RQ2) |

---

### 📊 Bias Table (Human vs LLM Means, Bias & MSE)

| **Attribute** | **Claude (Human / LLM / Bias / MSE)** | **GPT-4o** | **Gemini** | **o4-mini** |
|--------------|----------------------------------------|-----------|-----------|-----------|
| Guidance | 3.742 → 3.990 → +0.248 → 0.923 | 3.656 → 4.427 → +0.771 → 1.513 | 3.667 → 4.154 → +0.486 → 1.368 | 3.680 → 4.120 → +0.440 → 1.114 |
| Informativeness | 4.032 → 3.931 → −0.101 → 0.829 | 3.951 → 4.412 → +0.461 → 0.958 | 3.956 → 4.071 → +0.115 → 1.032 | 3.963 → 3.819 → −0.144 → 0.846 |
| Relevance | 4.520 → 4.574 → +0.054 → 0.999 | 4.478 → 4.867 → +0.389 → 0.780 | 4.484 → 4.886 → +0.401 → 0.880 | 4.487 → 4.917 → +0.431 → 0.804 |
| Safety | 4.734 → 4.852 → +0.118 → 0.521 | 4.714 → 4.932 → +0.218 → 0.451 | 4.716 → 4.924 → +0.208 → 0.550 | 4.716 → 4.967 → +0.251 → 0.534 |
| Empathy | 4.046 → 4.687 → +0.641 → 1.181 | 3.958 → 4.775 → +0.817 → 1.391 | 3.992 → 4.695 → +0.703 → 1.310 | 3.991 → 4.572 → +0.581 → 1.117 |
| Helpfulness | 3.972 → 4.399 → +0.427 → 0.946 | 3.869 → 4.538 → +0.669 → 1.130 | 3.896 → 4.643 → +0.747 → 1.354 | 3.888 → 4.362 → +0.474 → 0.912 |
| Understanding | 4.511 → 4.543 → +0.031 → 1.084 | 4.472 → 4.821 → +0.349 → 0.769 | 4.477 → 4.875 → +0.397 → 0.934 | 4.478 → 4.780 → +0.303 → 0.758 |

---

### 📌 Interpretation Summary

| Attribute Type | Typical Bias | Fixable? | Implication |
|----------------|-------------|---------|-------------|
| 🧠 Cognitive | Small (+0.25–0.46) | ✅ Yes | Calibration possible |
| 💛 Affective | High (+0.58–0.82) | ⚠️ Risky | Could cause false trust |
| 🚨 Safety/Relevance | Moderate (+0.18–0.39) but low reliability | ❌ No | Requires human oversight |

---

### ✅ Takeaway

> *LLM judges don’t just “drift randomly”—they show predictable over-optimism, especially in emotionally sensitive attributes like empathy and helpfulness. While cognitive scoring can be calibrated, affective and safety-critical assessments require stricter human supervision.*

