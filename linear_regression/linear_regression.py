import numpy as np
import matplotlib.pyplot as plt

SIZE = 50

def show_charts(true_weights, ai_weights):
    plt.title('AI for linear function.')
    plt.plot(true_weights, label='True weights')
    plt.plot(ai_weights, label='AI weights')

    plt.show()


def calc_weights(X, real_result, generation):
    ai_weights = np.random.randn(SIZE)
    learn_curve = 0.001 # Zavisi ot porqdyka na X
    progression = 10.0 # ?

    for _ in range(generation):
        print(ai_weights)
        predict_result = X.dot(ai_weights)
        delta = predict_result - real_result
        ai_weights = ai_weights - learn_curve * (X.T.dot(delta) + progression * np.sign(ai_weights))

    return ai_weights


def main():
    X = (np.random.random((SIZE, SIZE)) - 0.5) * 10
    true_weight = np.random.rand(SIZE)
    real_result = X.dot(true_weight) + np.random.randn(SIZE) / 2
    ai_weights = calc_weights(X, real_result, 1000)
    # for i in range(len(ai_weights)):
    #     print(ai_weights[i], true_weight[i])

    show_charts(true_weight, ai_weights)


if __name__ == '__main__':
    main()
