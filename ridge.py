#coding=utf-8
from linear_regression import LinearRegreession
from linear_regression import run_time
from linear_regression import load_data
from linear_regression import min_max_scale
from linear_regression import train_test_split
from linear_regression import get_r2

from numpy import ndarray

class Ridge(LinearRegreession):
    """脊回归类
        损失函数:
        L = (y - y_hat) ^ 2 + L2
        L = (y - W * X - b) ^ 2 + α * (W, b) ^ 2
        Get partial derivative of W:
        dL/dW = -2 * (y - W * X - b) * X + 2 * α * W
        dL/dW = -2 * (y - y_hat) * X + 2 * α * W
        Get partial derivative of b:
        dL/db = -2 * (y - W * X - b) + 2 * α * b
        dL/db = -2 * (y - y_hat) + 2 * α * b
        ----------------------------------------------------------------
        超参数:
            bias: b
            weights: W
            alpha: α
    """
    def __init__(self):
        super(Ridge, self).__init__()
        self.alpha = None

    def get_gradient_delta(self, Xi, yi):
        y_hat = self._predict(Xi)
        bias_grad_delta = yi - y_hat - self.alpha * self.bias
        weights_grad_delta = [(yi - y_hat) * Xij - self.alpha * wj
                              for Xij, wj in zip(Xi, self.weights)]
        return bias_grad_delta, weights_grad_delta

    def fit(self, X, y, lr, epochs, alpha, method="batch", sample_rate=1.0):
        self.alpha = alpha
        assert method in ("batch", "stochastic")
        if method == "batch":
            self.batch_gradient_descent(X, y, lr, epochs)
        if method == "stochastic":
            self.stochastic_gradient_descent(X, y, lr, epochs, sample_rate)

@run_time
def main():
    print("Tesing the performance of Ridge Regressor(stochastic)...")
    # Load data
    data, label = load_data()
    data = min_max_scale(data)
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=10)
    # Train model
    reg = Ridge()
    reg.fit(X=data_train, y=label_train, lr=0.001, epochs=1000, method="stochastic", sample_rate=0.5, alpha=1e-4)
    # Model evaluation
    get_r2(reg, data_test, label_test)
