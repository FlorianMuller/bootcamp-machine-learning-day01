import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR


def mse_thing(Xpill, Yscore):
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))

    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    # Model 1
    print("MyLR:    ", linear_model1.cost_(Xpill, Yscore) * 2)
    print("sklearn: ", mean_squared_error(Yscore, Y_model1))

    # Model 2
    print("MyLR:    ", linear_model2.cost_(Xpill, Yscore) * 2)
    print("sklearn: ", mean_squared_error(Yscore, Y_model2))


def plot_data_and_best_hypothesis(lr, x, y, xlabel=None, ylabel=None):
    lr.fit_(x, y)
    print("thetas", lr.thetas)

    # Given data
    plt.plot(x, y, "co", label="true data")

    y_hat = lr.predict_(x)
    # Prediction line
    plt.plot(x, y_hat, "--", color="lime", label="prediction")
    # Prediction data
    plt.plot(x, y_hat, "go")

    # Other
    # plt.title("Quantit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")

    plt.show()


def plot_cost_func(x, y, hue=0.6, brightness=1):
    lr = MyLR([0, 0])

    theta0_values = np.arange(80, 105, 5)
    saturation = np.linspace(0, 1, theta0_values.size)
    for t0, sat in zip(theta0_values, saturation):
        lr.thetas[0] = t0
        color = (hue, sat, brightness)

        # Draw one curve
        theta1_values = np.linspace(-100, 100, 1000)
        y_cost = np.zeros(theta1_values.shape)
        for i, t1 in enumerate(theta1_values):
            lr.thetas[1] = t1
            y_cost[i] = lr.cost_(x, y)

        plt.plot(theta1_values, y_cost, '-', color=color,
                 label=f"$J=(\\theta_0={t0}, \\theta_1)$")

    ax = plt.gca()
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3)

    plt.xlabel("$\\theta_1$")
    plt.ylabel("Cost function $J(\\theta_0,\\theta_1)$")
    plt.axis([-14, -4, 0, 150])

    plt.show()


def main():
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    mse_thing(Xpill, Yscore)

    plot_cost_func(Xpill, Yscore)

    lr = MyLR([1, 1], n_cycle=100000)
    plot_data_and_best_hypothesis(lr, Xpill, Yscore,
                                  xlabel="Quantity of blue pill",
                                  ylabel="Space driving score")


if __name__ == "__main__":
    main()
