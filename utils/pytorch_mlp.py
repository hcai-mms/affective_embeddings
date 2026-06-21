import numpy as np
import torch.nn as nn
import torch
from skorch import NeuralNet, NeuralNetClassifier

HIDDEN_LAYERS = [
            (64,), (128,), (256,), (512,), (1024,),
            (128, 64), (256, 128), (512, 256), (1024, 512),
            (256, 128, 64), (512, 256, 128), (1024, 512, 256)
        ]

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, complexity, activation='relu', final_activation=None, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_size = input_dim

        for hidden_size in HIDDEN_LAYERS[complexity]:
            layers.append(nn.Linear(prev_size, hidden_size))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'logistic':
                layers.append(nn.Sigmoid())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        if final_activation is not None:
            if final_activation == 'tanh':
                layers.append(nn.Tanh())
            elif final_activation == 'logistic':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Cannot append {final_activation} to model")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetBCE(NeuralNet):
    _estimator_type = "classifier"

    def fit(self, X, y, **fit_params):
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 2:
            raise ValueError(f"Expected y shape (n_samples, n_labels), got {y.shape}")

        self.classes_ = np.arange(y.shape[1])
        self.n_features_in_ = X.shape[1]

        return super().fit(X, y, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return super().get_loss(y_pred, y_true.float(), X=X, training=training)

    def predict_proba(self, X):
        probas = []
        for logits in self.forward_iter(X, training=False):
            probas.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(probas, axis=0)