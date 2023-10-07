import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlops.model import SimpleModel
from mlops.utils import model_save_inf, model_train

if __name__ == "__main__":
    SAVE_PATH = "./models/iris.pt"

    X, y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.20, random_state=99  # type: ignore
    )

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    model = SimpleModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model, _ = model_train(model, X_train, y_train, optimizer, criterion, 30)

    model_save_inf(model, SAVE_PATH)
