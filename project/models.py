import torch
import torch.nn as nn
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    """
    A standard Convolutional Neural Network backbone for 1x28x28 images.
    Returns a flattened feature vector.
    """
    def __init__(self, use_dropout=False, dropout_p=0.2):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout2d(p=self.dropout_p)

    def forward(self, x):
        x = self.relu(self.pool1(self.bn1(self.conv1(x))))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        if self.use_dropout:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

class LabelPredictor(nn.Module):
    """
    Takes the extracted features and predicts the 10-class label.
    """
    def __init__(self, input_dim=3136, hidden_dim=512, use_dropout=False, dropout_p=0.5):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)
        
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainClassifier(nn.Module):
    """
    Takes the extracted features and predicts the domain (source=0, target=1).
    """
    def __init__(self, input_dim=3136, hidden_dim=512):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, alpha):
        x = GradientReversalFunction.apply(x, alpha)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
