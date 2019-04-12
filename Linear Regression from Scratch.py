import numpy as np


def error_rate_for_parameters(data, m, b):
    sum_of_squared_error = 0

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        sum_of_squared_error += ((((m * x) + b) - y) ** 2)

    error_rate = sum_of_squared_error/(float(len(data)))
    return error_rate


def gradient_decent(data, initial_m, initial_b, learning_rate, number_of_iterations):
    m = initial_m
    b = initial_b
    
    for _ in range(number_of_iterations):
        m, b = step_gradient(data, m, b, learning_rate)
        
    return m, b


def step_gradient(data, m, b, learning_rate):
    m_gradient = 0
    b_gradient = 0

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]

        m_gradient += (((m*x) + b) - y) * x
        b_gradient += ((m*x) + b) - y
    
    m = m - ((learning_rate / len(data)) * m_gradient)
    b = b - ((learning_rate / len(data)) * b_gradient)

    return m, b


def main():
    data = np.genfromtxt('data.csv', delimiter=',')

    initial_m = 0
    initial_b = 0

    learning_rate = 0.0008
    number_of_iterations = 1000

    print(f"Starting with m as {initial_m} and b as {initial_b} we get error {error_rate_for_parameters(data, initial_m, initial_b)}")

    print("Running....")

    m, b = gradient_decent(data, initial_m, initial_b, learning_rate, number_of_iterations)

    print(f"After {number_of_iterations} iterations, m as {m} and b as {b} we get error {error_rate_for_parameters(data,m,b)}")


if __name__ == '__main__':
    main()
