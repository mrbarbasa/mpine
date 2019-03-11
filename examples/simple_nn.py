# Source:
# https://peterroelants.github.io/posts/neural-network-implementation-part01/

import matplotlib
matplotlib.use('TkAgg')

import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
np.random.seed(seed=13)


def print_versions():
    print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
    print('numpy: {}'.format(np.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('seaborn: {}'.format(sns.__version__))
# print_versions()


def f(x):
    # Generate the target values t from x with small Gaussian noise
    # so that the estimation won't be perfect.
    # This function represents the line that generates t without noise.
    # f(x) = 2x
    return x * 2


def nn(x, w):
    # Output function y = x * w
    return x * w


def loss(y, t):
    # MSE loss function
    return np.mean((t - y)**2)


###
# Plot the samples and regression line
###

# Vector of input samples
# 20 values sampled from uniform dist between 0 and 1
x = np.random.uniform(0, 1, 20)

# Generate the Guassian noise error for each sample in x
noise = np.random.randn(x.shape[0]) * 0.2
# Create targets using the formula: f(x) + N(0, 0.2)
t = f(x) + noise


def plot_samples():
    # Plot the target t versus the input x
    plt.figure(figsize=(5, 3))
    plt.plot(x, t, 'o', label='$t$')
    # Plot the initial line
    plt.plot([0, 1], [f(0), f(1)], 'b--', label='$f(x)$')
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$t$', fontsize=12)
    plt.axis((0, 1, 0, 2))
    plt.title('inputs (x) vs targets (t)')
    plt.legend(loc=2)
    plt.show()
# plot_samples()


###
# Plot the loss vs. the given weight w
###

# Vector of weights for which we want to plot the loss
ws = np.linspace(0, 4, num=100)  # weight values
# Create a vectorized function
vectorized_fn = np.vectorize(lambda w: loss(nn(x, w), t))
# Calculate the loss for each weight in ws
loss_ws = vectorized_fn(ws)


def plot_loss_function():
    plt.figure(figsize=(5, 4))
    plt.plot(ws, loss_ws, 'r--', label='loss')
    plt.xlabel('$weights$', fontsize=12)
    plt.ylabel('$MSE$', fontsize=12)
    plt.title('loss function with respect to $w$')
    plt.xlim(0, 4)
    plt.legend()
    plt.show()
# plot_loss_function()


###
# Plot gradient descent updates on the loss function
###

def gradient(w, x, t):
    """Gradient function to multiply the learning rate by.
    Note: y = nn(x, w) = x * w.

    Gradient: 2x(y_pred - y_true)
    """
    return 2 * x * (nn(x, w) - t)


def delta_w(w_k, x, t, learning_rate):
    """Full weight update function for a batch of samples."""
    return learning_rate * np.mean(gradient(w_k, x, t))


# Initial weight parameter
w = np.random.rand()
# Set the learning rate
learning_rate = 0.9

# Perform the gradient descent updates and print the weights and loss
# Number of gradient descent updates
nb_of_iterations = 4
# Keep track of weight and loss values; include initial values
w_loss = [(w, loss(nn(x, w), t))]

for i in range(nb_of_iterations):
    # Delta w update
    dw = delta_w(w, x, t, learning_rate)
    print(f'gradient at w({i}): {dw:.4f}')
    # Update the current weight parameter in the negative direction of the
    # gradient
    w = w - dw
    # Loss value
    loss_value = loss(nn(x, w), t)
    # Save weight and loss
    w_loss.append((w, loss_value))

# Print the final weight and loss
# Target weight value is around 2.0
for i in range(0, len(w_loss)):
    w = w_loss[i][0]
    l = w_loss[i][1]
    print(f'w({i}): {w:.4f}\tloss: {l:.4f}')


def plot_gradient_updates():
    # Visualise the gradient descent updates
    plt.figure(figsize=(6, 4))
    plt.plot(ws, loss_ws, 'r--', label='loss')  # Loss curve
    # Plot the updates
    for i in range(0, len(w_loss) - 1):
        w1, c1 = w_loss[i]
        w2, c2 = w_loss[i + 1]
        plt.plot(w1, c1, 'bo')
        plt.plot([w1, w2], [c1, c2], 'b-')
        plt.text(w1, c1 + 0.05, f'${i}$')
    plt.plot(w2, c2, 'bo', label='$w(k)$')
    plt.text(w2, c2 + 0.05, f'${i+1}$')
    # Show figure
    plt.xlabel('$w$', fontsize=12)
    plt.ylabel('$\\xi$', fontsize=12)
    plt.title('Gradient descent updates plotted on loss function')
    plt.xlim(0, 4)
    plt.legend(loc=1)
    plt.show()
# plot_gradient_updates()


###
# Plot the final results
###

w = np.random.rand()
nb_of_iterations = 10

for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)
    w = w - dw


def plot_final_result():
    # Note that there is no bias term as both lines pass through the origin
    # Plot the fitted line agains the target line
    plt.figure(figsize=(6, 4))
    # Plot the target t versus the input x
    plt.plot(x, t, 'o', label='$t$')
    # Plot the initial line
    plt.plot([0, 1], [f(0), f(1)], 'b--', label='$f(x)$')
    # plot the fitted line
    plt.plot([0, 1], [0 * w, 1 * w], 'r-', label='$y = w * x$')
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$t$', fontsize=12)
    plt.title('input vs target')
    plt.legend(loc=2)
    plt.ylim(0, 2)
    plt.xlim(0, 1)
    plt.show()
plot_final_result()
