# Multiple Linear Regression Engine

A vectorized implementation of Linear Regression using Gradient Descent.

## Key Technical Features

1. Vectorized Gradient Descent: Implemented weight updates using the identity: $\nabla_\theta J(\theta) = \frac{1}{n} X^T(X\theta - y)$.

2. Bias Integration: Automatic feature augmentation for intercept handling.

3. L1/L2 Regularization: Integrated weight penalties in the optimizer to prevent overfitting.

4. Evaluation Suite: Built-in $R^2$ Score and MSE (Mean Squared Error) calculation.

5. Unit Tested: Includes test_math.py to verify matrix dimensions and convergence logic.


## Benchmarking (Custom vs. Scikit-Learn)

To validate the engine, the model was tested against synthetic data with high noise (range: $[-10, 10]$).

| Metric | Custom Model | Sklearn |
| :--- | :---: | ---: |
| Train R2 | 0.9774 | 0.9775 |
| Test R2 | 0.8912 | 0.8999 |
|Parity|	99.9%|	Reference|

## Project Structure

### 📂 Directory Layout
```text
.
├── LinearRegression/
│   ├── main.py
│   └── src/
│       ├── __init__.py
│       ├── model.py
│       ├── optimizer.py
│       ├── utils.py
│       └── test_math.py
├── .gitignore
└── README.md
```
