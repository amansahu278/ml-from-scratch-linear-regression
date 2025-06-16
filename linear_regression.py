import torch

class LinearRegression:
    def __init__(self, closed_repr=False, learning_rate=0.01, n_iterations=1):
        self.closed_repr = closed_repr
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        print(f"X.shape: {X.shape}")
        print(f"Y shape: {y.shape}")

        n_samples, n_features = X.shape
        if y.dim() == 1:
            n_outputs = 1
            y = y.reshape(-1, 1) # (n_samples, 1)
        else:
            n_outputs = y.shape[1] # (n_samples, n_outputs)

        self.w = torch.zeros((n_features, n_outputs), dtype=torch.float32)
        self.b = torch.zeros(n_outputs, dtype=torch.float32) # Scalar

        print(f"Weights shape: {self.w.shape}")
        print(f"Bias shape: {self.b.shape}")

        if self.closed_repr:
            # Closed form solution
            # y = wx, note: no bias
            intermediate = X.T @ X
            self.w = torch.linalg.inv(intermediate) @ X.T @ y
            self.b = torch.zeros(n_outputs, dtype=torch.float32)
        else:
            for i in range(self.n_iterations):
                y_pred = self.predict(X)
                mse_loss = ((y - y_pred) ** 2).mean(dim=0)
                print(f"Iteration: {i}, MSE Loss: {mse_loss}")

                intermediate = (y - y_pred) / n_samples
                grad_w = -2 * X.T @ intermediate
                grad_b = -2 * intermediate.mean(dim=0) # Scalar
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

    def predict(self, X):
        return X @ self.w + self.b
    
if __name__ == "__main__":

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = torch.tensor(X_train, dtype=torch.float32) # (n_samples, n_features)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32) # (n_samples,)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model_closed = LinearRegression(closed_repr=True)
    model_gd = LinearRegression(learning_rate=0.01, n_iterations=10)

    model_closed.fit(X_train, y_train)
    model_gd.fit(X_train, y_train)

    y_closed_pred = model_closed.predict(X_test)
    print(f"y_pred shape: {y_closed_pred.shape}")
    y_gd_pred = model_gd.predict(X_test)

    print(f"y_pred - y_test: {(y_closed_pred - y_test).shape}")

    # NOTE
    # y_test is (n_samples,), y_pred is (n_samples, 1)
    # y_pred - y_test will be (n_samples, n_samples) due to broadcast, this .squeeze()


    mse1 = ((y_closed_pred.squeeze() - y_test) ** 2).mean().item()
    mse2 = ((y_gd_pred.squeeze() - y_test) ** 2).mean().item()

    print(f"Closed representation mse loss: {mse1}")
    print(f"Gradient Descent mse loss: {mse2}")