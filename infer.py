import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from mlops.model import SimpleModel
from mlops.utils import model_predict, model_load_inf
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X, y = load_iris(as_frame=True, return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X.values, y.values, test_size=0.20, random_state=99  # type: ignore
    )

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    model_inf = SimpleModel()
    model_load_inf(model_inf, "./models/iris.pt")
    y_pred = model_predict(model_inf, X_test)

    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
