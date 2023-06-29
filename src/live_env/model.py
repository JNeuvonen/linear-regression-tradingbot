from torch import nn, device
import torch


class LinearModel(nn.Module):
    def __init__(self, input_params):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_params, 1)

    def forward(self, x):
        return self.linear(x)


def initiate_model(input_params_count):
    device = torch.device("cpu")
    model = LinearModel(input_params_count)
    model.load_state_dict(torch.load(
        "model.pt", map_location=torch.device('cpu')))
    model.to(device)
    return model
