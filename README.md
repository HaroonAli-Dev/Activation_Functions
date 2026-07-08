# 🧠 Activation Functions in Deep Learning

> A comprehensive visual and mathematical guide to the most important activation functions used in neural networks.

---

## 📌 What is an Activation Function?

An **activation function** is a mathematical function applied to the output of each neuron in a neural network. It introduces **non-linearity**, enabling the network to learn and model complex, real-world patterns that a purely linear transformation cannot capture.

> **Without activation functions** → a deep neural network is just a linear regression model, regardless of how many layers it has.

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `activation_functions.ipynb` | Main Jupyter notebook with full implementation and visualizations |
| `README.md` | This documentation file |

---

## 📦 Requirements

```bash
pip install numpy matplotlib pandas scipy
```

Or with conda:
```bash
conda install numpy matplotlib pandas scipy
```

---

## 🔢 Activation Functions Covered

### 1️⃣ Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| Property | Value |
|----------|-------|
| Output Range | **(0, 1)** |
| Zero-Centered | ❌ No |
| Differentiable | ✅ Yes |
| Vanishing Gradient | ❌ Severe |

**When to use:**
- Output layer for **binary classification** tasks
- Probability estimation

**Problems:**
- Saturates at both ends → **vanishing gradient**
- Not zero-centered → slow convergence
- Computationally more expensive than ReLU

---

### 2️⃣ Tanh (Hyperbolic Tangent)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

| Property | Value |
|----------|-------|
| Output Range | **(-1, 1)** |
| Zero-Centered | ✅ Yes |
| Differentiable | ✅ Yes |
| Vanishing Gradient | ⚠️ Mild |

**When to use:**
- Hidden layers in **RNNs and LSTMs**
- Preferred over Sigmoid for hidden layers

**Advantage over Sigmoid:** Zero-centered output → faster convergence, 4× stronger gradients.

---

### 3️⃣ ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

| Property | Value |
|----------|-------|
| Output Range | **[0, ∞)** |
| Zero-Centered | ❌ No |
| Computational Cost | ✅ Very Low |
| Vanishing Gradient | ✅ No (for x > 0) |

**When to use:**
- Default choice for **hidden layers in CNNs and feedforward networks**
- When computational efficiency matters

**Problems:**
- **Dying ReLU**: Neurons stuck at 0 with gradient = 0
- Unbounded above → possible exploding activations

---

### 4️⃣ Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

where $\alpha$ (typically 0.01) is a small slope for negative inputs.

| Property | Value |
|----------|-------|
| Output Range | **(-∞, ∞)** |
| Dying Neurons | ✅ Fixed |
| Vanishing Gradient | ✅ No |

**When to use:**
- When neurons are dying in standard ReLU networks
- **PReLU** variant: learns $\alpha$ during training

---

### 5️⃣ ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

| Property | Value |
|----------|-------|
| Output Range | **(-α, ∞)** |
| Smooth | ✅ Yes |
| Noise Robust | ✅ Yes |
| Vanishing Gradient | ✅ No |

**When to use:**
- Deep networks where smoother gradients are needed
- When negative outputs have semantic meaning

---

### 6️⃣ Softmax

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

| Property | Value |
|----------|-------|
| Output Range | **(0, 1), sums to 1** |
| Applied To | Vector (not element-wise) |
| Use Case | Output layer only |

**When to use:**
- **Multi-class classification** output layer (exclusive categories)
- Always paired with **Cross-Entropy Loss**

> ℹ️ Softmax is a vector operation — it turns a vector of raw scores (logits) into a probability distribution.

---

### 7️⃣ Swish

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

| Property | Value |
|----------|-------|
| Output Range | **(-∞, ∞)** |
| Monotonic | ❌ No (non-monotonic) |
| Origin | Google Brain (NAS, 2017) |
| Vanishing Gradient | ✅ No |

**When to use:**
- **EfficientNet, MobileNetV3**
- Deep networks where ReLU is the default but performance can be improved

**Unique property:** Non-monotonic — has a small dip below 0 for negative inputs.

---

### 8️⃣ GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

| Property | Value |
|----------|-------|
| Output Range | **(-∞, ∞)** |
| Smooth | ✅ Yes |
| Used In | BERT, GPT, ViT, etc. |
| Vanishing Gradient | ✅ No |

**When to use:**
- **Transformer-based models** (BERT, GPT-2, GPT-3, ViT)
- NLP and Vision tasks with self-attention architectures

**Note:** The exact formula uses the error function (erf). A fast approximation using `tanh` is widely used in practice.

---

## 📊 Quick Comparison Table

| Function     | Output Range  | Zero-Centered | Vanishing Grad | Dying Neurons | Computational Cost | Best For                     |
|--------------|---------------|---------------|----------------|---------------|--------------------|------------------------------|
| Sigmoid      | (0, 1)        | ❌            | ❌ Severe      | No            | Medium             | Binary output layer          |
| Tanh         | (-1, 1)       | ✅            | ⚠️ Mild        | No            | Medium             | RNN/LSTM gates               |
| ReLU         | [0, ∞)        | ❌            | ✅ No          | ❌ Yes        | Very Low           | CNNs, hidden layers          |
| Leaky ReLU   | (-∞, ∞)       | ❌            | ✅ No          | ✅ Fixed      | Very Low           | When ReLU dies               |
| ELU          | (-α, ∞)       | ≈ ✅          | ✅ No          | ✅ Fixed      | Low-Medium         | Better than ReLU             |
| Softmax      | (0,1) sum=1   | N/A           | N/A            | N/A           | Medium             | Multi-class output           |
| Swish        | (-∞, ∞)       | ❌            | ✅ No          | ✅ No         | Medium             | EfficientNet, deep nets      |
| GELU         | (-∞, ∞)       | ❌            | ✅ No          | ✅ No         | High               | Transformers (BERT, GPT)     |

---

## 🎯 Decision Guide — Which Activation to Use?

```
OUTPUT LAYER
├── Binary Classification    → Sigmoid
├── Multi-class              → Softmax
└── Regression               → None (linear)

HIDDEN LAYERS
├── General / CNNs           → ReLU  (fast default)
├── Dying neurons issue      → Leaky ReLU or ELU
├── Transformers / NLP       → GELU
├── Deep networks            → Swish
└── RNNs / LSTMs             → Tanh + Sigmoid
    └── ⚠️ Avoid Sigmoid/Tanh in very deep feedforward nets
```

---

## 🔥 Key Problems & Solutions

| Problem                  | Activation to Use       |
|--------------------------|-------------------------|
| Vanishing gradient       | ReLU, ELU, Swish, GELU  |
| Dying ReLU neurons       | Leaky ReLU, ELU         |
| Need probability output  | Sigmoid / Softmax       |
| Training transformers    | GELU                    |
| Need sparsity            | ReLU                    |
| RNN memory gates         | Tanh + Sigmoid          |

---

## 📈 Notebook Contents

The `activation_functions.ipynb` notebook includes:

1. **Mathematical formulas** for each function and its derivative
2. **Function plots** (shape visualization)
3. **Derivative plots** (gradient analysis)
4. **Side-by-side comparison** of all 8 functions
5. **Gradient comparison** — visualizing the vanishing gradient problem
6. **Output distribution analysis** — effect on neuron outputs
7. **Comprehensive summary table** with pandas
8. **Decision guide** for choosing the right activation

---

## 🚀 How to Run

```bash
# Open the notebook
jupyter notebook activation_functions.ipynb

# Or with JupyterLab
jupyter lab activation_functions.ipynb
```

Each cell is self-contained and can be run independently. Run all cells top-to-bottom for the full analysis.

---

## 📚 References

- [Goodfellow et al. (2016) — Deep Learning (Book)](https://www.deeplearningbook.org/)
- [Ramachandran et al. (2017) — Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941)
- [Hendrycks & Gimpel (2016) — Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
- [Clevert et al. (2015) — Fast and Accurate Deep Network Learning by Exponential Linear Units (ELU)](https://arxiv.org/abs/1511.07289)
- [Maas et al. (2013) — Rectifier Nonlinearities Improve Neural Network Acoustic Models (Leaky ReLU)](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

---

*Made with 🐍 Python, 📊 Matplotlib, and 🔢 NumPy*
