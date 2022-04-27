from torch import _nn

class NeuralNetwork(_nn.Module):
    def __init__(self, n_inputs, n_outputs=1):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = _nn.Sequential(
            _nn.Linear(n_inputs, 512),
            _nn.ReLU(),
            _nn.Linear(512, 512),
            _nn.ReLU(),
            _nn.Linear(512, n_outputs),
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
