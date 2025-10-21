### 📊 Human Evaluation Results (RQ1 — Response Generation Performance)

The table below presents human-rated evaluation scores (scale: 1–5) for each LLM across **seven therapeutic attributes** over **1,000 conversations**. Ratings were provided by trained human experts, and the overall average score is included in the final column.

| **Model**            | **Source** | **Guidance** | **Informativeness** | **Relevance** | **Safety** | **Empathy** | **Helpfulness** | **Understanding** | **Avg** |
|----------------------|-----------|--------------|----------------------|--------------|-----------|------------|----------------|------------------|--------|
| **GPT-4o**           | **Closed** | **4.51**     | **4.76**             | **4.89**     | **4.96**  | **4.60**   | **4.72**       | **4.89**         | **4.76** |
| Gemini-2.0-Flash     | Closed     | 4.41         | 4.72                 | 4.84         | 4.95      | 4.30       | 4.49           | 4.85             | 4.65   |
| GPT-4o-Mini          | Closed     | 4.30         | 4.64                 | 4.82         | 4.95      | 4.31       | 4.55           | 4.84             | 4.63   |
| *LLaMA-3.1-8B*       | *Open*     | *4.07*       | *4.51*               | *4.76*       | *4.89*    | *4.36*     | *4.42*         | *4.78*           | *4.54* |
| DeepSeek-LLaMA-8B    | Open       | 3.72         | 3.92                 | 4.50         | 4.76      | 4.16       | 3.87           | 4.49             | 4.20   |
| Qwen-2.5-7B          | Open       | 3.89         | 4.08                 | 4.39         | 4.55      | 4.01       | 4.13           | 4.38             | 4.20   |
| Claude-3.5-Haiku     | Closed     | 3.74         | 4.03                 | 4.53         | 4.79      | 3.82       | 3.81           | 4.55             | 4.18   |
| DeepSeek-Qwen-7B     | Open       | 3.60         | 3.88                 | 4.45         | 4.72      | 4.25       | 3.80           | 4.47             | 4.16   |
| Qwen-3-4B            | Open       | 3.07         | 3.32                 | 4.08         | 4.46      | 3.62       | 3.20           | 4.07             | 3.64   |

✅ **Bold** = best-performing model overall (including closed-source)  
✅ *Italicized* = best open-source model

---

### What this shows

This table corresponds to the results from **RQ1** and establishes the **human rating baseline**:

| Attribute | Meaning |
|----------|---------|
| **Guidance** | Ability to provide actionable steps or direction |
| **Informativeness** | Depth and richness of content |
| **Relevance** | Alignment with user’s expressed concern |
| **Safety** | Avoidance of harmful or risky suggestions |
| **Empathy** | Emotional attunement and recognition of user’s feelings |
| **Helpfulness** | Perceived usefulness from a user-support perspective |
| **Understanding** | Grasp of intent and context in the conversation |

---

### 📎 Key Insights

- **GPT-4o** is the top-performing model across all metrics, with the highest average score (**4.76/5**).
- **Gemini-2.0-Flash** and **GPT-4o-Mini** rank closely behind.
- Among **open-source models**, *LLaMA-3.1-8B* is the strongest performer, scoring an average of *4.54*.
- Lower-capacity open models like **Qwen-3-4B** trail behind, particularly in guidance, empathy, and helpfulness.
- Performance differences between cognitive attributes (Guidance, Informativeness, Relevance, Safety) and affective ones (Empathy, Helpfulness, Understanding) inform deeper analysis in later sections (RQ2, RQ3).
