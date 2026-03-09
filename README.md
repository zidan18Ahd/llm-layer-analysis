# LLM Layer Analysis: When Does GPT-2 Decide the Answer?

This project explores a simple but interesting question:

**At which layer of a transformer model does the correct answer actually appear?**

Large Language Models often feel like black boxes. You give them a prompt, they produce an answer — but what happens internally before that final prediction is made?

In this experiment, we use a simple interpretability technique inspired by the **logit lens** to inspect predictions across transformer layers in **GPT-2**.

---

## Idea

Instead of looking only at the final output of the model, we probe **every transformer layer**.

At each layer we:

1. Take the hidden state of the last token
2. Project it through the model’s output layer (`lm_head`)
3. Convert the logits to probabilities
4. Track the probability of the correct answer

This lets us see **how the model's prediction evolves during computation**.

---

## Example

Prompt:

```
The capital of Italy is
```

Observation:

The probability of the token **"Rome"** stays extremely small across most layers and only rises sharply in the **final transformer layer**.

This suggests that earlier layers are focused on **building contextual representations**, while the final layer maps those representations to vocabulary tokens.

---

## Sample Result

Layer-wise probability of the correct token:

```
Layer 0:  P(Rome) = 0.000015
Layer 1:  P(Rome) = ~0
Layer 2:  P(Rome) = ~0
...
Layer 11: P(Rome) = ~0
Layer 12: P(Rome) = 0.157
```

The correct answer emerges **very late in the network**.

---

## Prompts Tested

The experiment was repeated on several factual prompts:

| Prompt | Expected Answer |
|------|------|
| The capital of Italy is | Rome |
| The capital of France is | Paris |
| The capital of Germany is | Berlin |
| The capital of Spain is | Madrid |
| The largest planet in the solar system is | Jupiter |
| The fastest land animal is | cheetah |
| The author of Hamlet is | Shakespeare |
| The theory of relativity was proposed by | Einstein |
| The tallest mountain in the world is | Everest |
| The chemical symbol for water is | H |

For most prompts, the probability of the correct answer **remains near zero until the final transformer layer**.

---

## Visualization

Each prompt produces a plot showing:

```
Layer number  →  Probability of correct token

```

---

## Code

The core experiment is implemented using **Hugging Face Transformers**.

Key steps:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

text = "The capital of Italy is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states
lm_head = model.lm_head.weight
```

Then we project intermediate hidden states to vocabulary space to inspect predictions.

---

## Repository Structure

```
llm-layer-analysis
│
├── experiments
│   └── logit_lens.py
│
├── plots
│   ├── Rome_layer_plot.png
│   ├── Paris_layer_plot.png
│   ├── Berlin_layer_plot.png
│   └── ...
│
└── README.md
```

---

## References

1. Vaswani et al., 2017 — *Attention Is All You Need*
2. Radford et al., 2019 — *Language Models are Unsupervised Multitask Learners*
3. Nostalgebraist, 2020 — *Interpreting GPT: The Logit Lens*
4. Elhage et al., 2021 — *A Mathematical Framework for Transformer Circuits*

---

## Blog Post

Full explanation of the experiment:

https://lnkd.in/gM8FsBTN
---

## Contact

**Zidan Ahmed**

Email:  
zidan18za@gmail.com

LinkedIn:  
https://www.linkedin.com/in/zidan-ahmed-214134298

GitHub:  
https://github.com/zidan18Ahd

---

## License

This project is for educational and research purposes.
