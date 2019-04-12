import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    data = np.genfromtxt('data.csv', delimiter=',')

    # Every value of x is stored as a array [x] because scikit needs x as a 2D array.
    x = [[value] for value in data[:, 0]]
    y = data[:, 1]

    # Split the dataset into the training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train, y_train)
    y_prediction = linear_regressor.predict(x_test)

    m = linear_regressor.coef_[0]
    b = linear_regressor.intercept_

    print(f"We get m as {m} and b as {b}.")
    print(f"Mean squared error - {mean_squared_error(y_test, y_prediction)}.")


if __name__ == '__main__':
    main()
