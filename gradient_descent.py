import numpy as np


def gradient_descent(x, y, learning_rate, iterations):
    a = np.column_stack((np.ones(len(y)), x))
    theta = np.zeros(a.shape[1])
    
    # MSE loss: L(theta) = (1/m) * ||A @ theta - y||^2
    # Gradient: ∇L(theta) = 2 * (A.T @ A / m @ theta - A.T @ y / m)
    # Precompute G = A.T @ A / m and c = A.T @ y / m.
    # Then ∇L(theta) = 2 * (G @ theta - c).
    gram = a.T @ a / len(y)
    cross = a.T @ y / len(y)

    initial_loss = float(np.mean(y ** 2))
    losses = np.empty(iterations + 1)
    losses[0] = initial_loss

    for step in range(1, iterations + 1):
        theta -= learning_rate * 2 * (gram @ theta - cross)
        losses[step] = theta @ gram @ theta - 2 * cross @ theta + initial_loss
    losses[-1] = np.mean((a @ theta - y) ** 2)

    return theta, losses
