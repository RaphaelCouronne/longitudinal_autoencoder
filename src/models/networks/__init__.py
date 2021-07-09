import torch


class SineActivation(torch.nn.Module):

    def __init__(self):
        """
        Sine activation function
        """
        super(SineActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class MetaActivation(torch.nn.Module):

    def __init__(self, name):
        """
        initialize among used activations
        """
        super(MetaActivation, self).__init__()
        assert name in ['identity', 'tanh', 'relu', 'lrelu', 'prelu', 'celu', 'selu', 'softplus', 'sigmoid', 'sine']
        self.name = name
        if self.name == 'identity':
            self.activation = torch.nn.Identity()
        elif self.name == 'tanh':
            self.activation = torch.nn.Tanh()
        elif self.name == 'relu':
            self.activation = torch.nn.ReLU()
        elif self.name == 'lrelu':
            self.activation = torch.nn.LeakyReLU()  # fixed default slope
        elif self.name == 'prelu':
            self.activation = torch.nn.PReLU()
        elif self.name == 'celu':
            self.activation = torch.nn.CELU()
        elif self.name == 'selu':
            self.activation = torch.nn.SELU()
        elif self.name == 'softplus':
            self.activation = torch.nn.Softplus()
        elif self.name == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif self.name == 'sine':
            self.activation = SineActivation()
        else:
            raise ValueError(" Activation not recognized")

    def forward(self, x):
        return self.activation(x)
