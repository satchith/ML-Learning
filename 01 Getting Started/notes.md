# 01 - Linear Regression: Normal Equation & Gradient Descent

**Reference:** Géron, *Hands-On ML* Ch. 4 | **Plan:** Week 1 (what a network is; gradient descent)

## What I set out to do
Fit a simple linear model `y = 4 + 3x + noise` three different ways and check they all agree:
1. the **normal equation** (closed-form solution),
2. **batch gradient descent** implemented from scratch,
3. scikit-learn's `LinearRegression` (as a sanity check).

Then visualise how the **learning rate** changes the way gradient descent converges.

## What I learned
- All three methods land on the same parameters (intercept ≈ 3.69, slope ≈ 3.33), which confirms gradient descent is just an iterative way of reaching the same
  minimum the normal equation finds directly.
- The bias term is handled by adding a column of 1s to X, so `theta_0` gets a constant feature `x0 = 1`.
- The gradient descent update is `theta <- theta - alpha * gradient`, where the gradient of the MSE loss is `(2/m) * X.T @ (X @ theta - y)`.
- **Learning rate matters a lot:** at `eta = 0.02` convergence is slow, at `eta = 0.1` it's smooth and quick, and at `eta = 0.5` the steps overshoot and
  the lines bounce around before settling.
- The big-picture idea: **learning = descending a loss landscape.** Each GD step moves the parameters downhill on the loss surface toward the minimum.