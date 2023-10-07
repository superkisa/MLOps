import torch
import torch.nn as nn
from tqdm import tqdm


def model_train(
    model: nn.Module,
    X_train,
    y_train,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    epochs=50,
):
    losses = []

    for i in (pbar := tqdm(range(epochs))):
        optimizer.zero_grad()

        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        pbar.set_description(f"Epoch #{i + 1:2}, loss: {loss.item():10.8f}")

        loss.backward()
        optimizer.step()
    return model, losses


def model_predict(model: nn.Module, X_test) -> list[int]:
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    preds = []
    with torch.no_grad():
        for val in X_test:
            y_hat = model.forward(val)
            preds.append(y_hat.argmax().item())
    if was_training:
        model.train()
    return preds


def model_save_inf(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def model_load_inf(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))
