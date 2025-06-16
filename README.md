# ml-from-scratch-linear-regression
# 📈 Linear Regression From Scratch

This project is a **from-scratch implementation** of classic linear regression using **pure PyTorch**. It is part of my ongoing GitHub series: [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch).

> 🎯 Goal: To deeply understand and re-implement linear regression without relying on high-level libraries like scikit-learn.

---

## 📚 Features

- Gradient descent optimizer
- Optional closed-form (normal equation) solution
- Supports single and multi-output regression

---

## 🏗️ Project Structure

```bash
.
├── linear_regression.py   # Main model class and training script
└── README.md
```

---

## 🚀 Usage

1. **Install dependencies**  
   Requires: `torch`, `scikit-learn`, `numpy`

   ```bash
   pip install torch scikit-learn numpy
   ```

2. **Run the script**

   ```bash
   python linear_regression.py
   ```

   The script will:
   - Load a regression dataset from scikit-learn
   - Train the model using both closed-form and gradient descent
   - Print MSE loss for both methods

---

## 📝 Example Output

```
X.shape: torch.Size([353, 10])
Y shape: torch.Size([353])
Weights shape: torch.Size([10, 1])
Bias shape: torch.Size([1])
Iteration: 0, MSE Loss: ...
...
Closed representation mse loss: ...
Gradient Descent mse loss: ...
```

---

## 🧩 Extending

- Add L2 regularization or mini-batch support

**Part of the [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch) series.**