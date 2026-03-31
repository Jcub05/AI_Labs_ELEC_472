import numpy as np
import matplotlib.pyplot as plt
# Let's assume a cost function for a model is: y = 3*x^2 - 2*x + 5
# To use gradient descent for minimizing this function over x
# (finding x such that y is minimum), let's assume an initial guess
# of x = 100 and a learning rate of 0.01
# Since the original function is y = 3*x^2 - 2*x + 5, then:
# dy/dx = 6*x – 2
# Therefore:
alpha = 0.01 # learning rate
x = 100 # initial guess
x_all = [] # to store x values
n_iterations = 500 # number of iterations
# Performing x = x – alpha*dy/dx for 500 iterations
for i in range(n_iterations):
    x = x - alpha * (6 * x - 2)
    x_all.append(x)
# Convert list to NumPy array for easier manipulation
x_all = np.array(x_all)
# Define the original function over a range of n values
n = np.arange(-100, 100, 0.001)
y = 3 * n**2 - 2 * n + 5
# Calculate the function values at the gradient descent estimates
y_sol = 3 * x_all**2 - 2 * x_all + 5
# Plot the original function
plt.figure()
plt.plot(n, y, 'b', label='Original function')
# Now, plotting the solutions estimated by GD
plt.plot(x_all, y_sol, '-or', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.show()