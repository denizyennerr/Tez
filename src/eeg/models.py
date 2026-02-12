import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# KAN LINEAR MODEL

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Creating the spline grid
        h = (grid_range[1] - grid_range[0]) / grid_size # step size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1) # expand to the number of features
            .contiguous() # contiguous tensor
        )
        self.register_buffer("grid", grid) # save the grid in the model
        
        # Learnable parameters
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # base weight matrix
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order) # spline weight matrix
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features) # spline scaler
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters() 

    # Reset parameters

    def reset_parameters(self):
        # Initialize the base weight matrix
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Initialize the spline weights with small random noise
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                ) # Small random values centered at 0
                * self.scale_noise
                / self.grid_size
            ) # Convert noise to spline coefficients
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], # Use middle part of the grid to avoid boundary effects
                    noise, # Small random values centered at 0
                ) # Convert noise to spline coefficients
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline) # Initialize the spline scaler

    # B-splines function

    def b_splines(self, x: torch.Tensor):
        # Check if the input is valid
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Recursive Construction of B-splines
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    # Converting Curves to Coefficients

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # Forward pass

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        # Base Transformation           
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Spline Transformation
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    # Adaptive Grid Update

    @torch.no_grad() # no gradient computation
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x) # Current basis functions
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight # Current spline coefficients
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)
        # compute the current spline outputs before updating the grid

        # Sort the input values

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

        # Regularization Loss

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1) # L1 regularization on the spline weights
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation # Probability distribution
        regularization_loss_entropy = -torch.sum(p * p.log()) # Entropy regularization
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        ) # Regularization loss

        # KAN Multi-Layer network

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    # Forward pass

    def forward(self, x: torch.Tensor, update_grid=False):
        # Forward pass through each layer
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x) # Update the grid if needed
            x = layer(x) # Forward pass through the layer
        return x # Return the output

    # Aggregated Regularization Loss
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# KAN Seizure Detector Wrapper

class KANSeizureDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], grid_size=5, dropout=0.2):
        super().__init__()
        # Input layer -> Hidden Layers -> 1 Output (Binary Classification)
        self.architecture = [input_dim] + hidden_layers + [1]
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        # KAN layers
        self.kan = KAN(
            layers_hidden=self.architecture,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )

    def forward(self, x, update_grid=False):
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Apply dropout before KAN layers to prevent overfitting
        x = self.dropout(x)
        logits = self.kan(x, update_grid=update_grid)
        return logits

