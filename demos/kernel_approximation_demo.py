import matplotlib.pyplot as plt
import seaborn as sns
from torchkernels.feature_maps import (
    LaplacianORF, LaplacianRFF,
    GaussianORF, GaussianRFF,
    MaternORF, MaternRFF,
    ExpPowerORF, ExpPowerRFF
    )
from torchkernels.kernels.radial import laplacian, gaussian, exp_power, matern
import numpy as np
import torch

class SubplotManager:
    def __init__(self, rows, cols, kernels):
        self.rows = rows
        self.cols = cols
        self.index = 0  # To track subplot index
        self.kernels = kernels  # List of kernels
        self.fig = plt.figure(figsize=(12, 6))  # Create figure

        # Set Seaborn style and Husl (Hus1) color palette
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", cols)  # 1 color per kernel (column)
        self.scatter_colors = sns.color_palette("icefire", cols) # Different color palette for scatter

        # Apply LaTeX-like styling without full LaTeX rendering
        plt.rcParams.update({
            "text.usetex": False,                 # Disable full LaTeX (optional for simplicity)
            "font.family": "serif",               # Use a serif font (similar to LaTeX math font)
            "mathtext.fontset": "cm",             # Use Computer Modern, the default LaTeX math font
            "mathtext.rm": "serif",               # Use serif font for regular text in math mode
        })

    def __enter__(self):
        return self  # Allow using 'with' statement

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._add_vertical_lines()  # Add lines before showing
        plt.tight_layout()
        plt.show()

    def next_subplot(self, row, col, title=""):
        """Move to the next subplot based on row and column index."""
        plt.subplot(self.rows, self.cols, row * self.cols + col + 1)
        plt.title(title)
        # plt.grid(True)

    def _add_vertical_lines(self):
        """Add vertical separator lines evenly between columns."""
        for i in range(1, self.cols):  # Add lines at exact column divisions
            x_pos = i / self.cols +0.01/i # Normalized x position
            self.fig.lines.append(plt.Line2D([x_pos, x_pos], [0, 1], 
                                             transform=self.fig.transFigure, 
                                             color='grey', linestyle=':', lw=1, alpha=0.5))


def kernel_approximation(input_dim = 32, num_features = int(3e4), length_scale = 1., bias_term = False, device = None, 
                         kernel = "Laplace", approx = "RFF", nu = 1.5, alpha = 0.7):
    assert kernel in ["Gaussian", "Laplacian", "Matern", "ExpPower"]
    if approx == "RFF":
        K_func_approx = {"Laplacian": LaplacianRFF, "Gaussian": GaussianRFF, "Matern": MaternRFF, "ExpPower": ExpPowerRFF}
    else: 
        K_func_approx = {"Laplacian": LaplacianORF, "Gaussian": GaussianORF, "Matern": MaternORF, "ExpPower": ExpPowerORF}
    
    K_func_exact = {"Laplacian": laplacian, "Gaussian": gaussian, "Matern": matern, "ExpPower": exp_power}
    N = int(20)
    x1_x2_norm = np.linspace(0, 5, N)
    K_mat_exact = []
    K_mat_approx = []
    if kernel == "ExpPower":
        feature_map = K_func_approx[kernel](num_features=num_features, input_dim=input_dim, length_scale=length_scale, alpha=alpha, bias_term=bias_term)
    elif kernel == "Matern":
        feature_map = K_func_approx[kernel](num_features=num_features, input_dim=input_dim, length_scale=length_scale, nu=nu, bias_term=bias_term)
    else:
        feature_map = K_func_approx[kernel](num_features=num_features, input_dim=input_dim, length_scale=length_scale, bias_term=bias_term)

    for i, x_ in enumerate(x1_x2_norm):
        x1 = torch.randn(input_dim)
        x1_x2_u = torch.randn(input_dim)
        x1_x2 = x1_x2_u/np.linalg.norm(x1_x2_u)*x_
        x2 = x1 + x1_x2
        x1 = x1.reshape(1,-1)
        x2 = x2.reshape(1,-1)
        Phi_x1 = feature_map(x1)
        Phi_x2 = feature_map(x2)
        K_mat_approx.append((Phi_x1@Phi_x2.T)[0].item())
        if kernel == "ExpPower":
            K_mat_exact.append(K_func_exact[kernel](x1, x2, length_scale=length_scale, alpha=alpha)[0].item())
        elif kernel == "Matern":
            K_mat_exact.append(K_func_exact[kernel](x1, x2, length_scale=length_scale, nu=nu)[0].item())
        else:
            K_mat_exact.append(K_func_exact[kernel](x1, x2, length_scale=length_scale)[0].item())
    return K_mat_exact, K_mat_approx, x1_x2_norm

def create_kernel_str(kernel:str, nu:float = 1.5, alpha:float = 0.7):
    assert kernel in ["Laplacian", "Gaussian", "Matern", "ExpPower"]
    if kernel == "ExpPower":
        return r"ExpPower ($\alpha = %.2f$)" % alpha
    elif kernel == "Matern":
        return r"Matérn ($\nu = %.2f$)" % nu
    else:
        return kernel

kernels = ["Gaussian", "Laplacian", "Matern", "ExpPower"]
nu = 1.5
alpha = 0.7
approximations = ["RFF", "ORF"]
kernel_str = {kernel: create_kernel_str(kernel, nu, alpha) for kernel in kernels}
with SubplotManager(rows=2, cols=4, kernels=kernels) as manager:
    for col, kernel in enumerate(kernels):  # Each kernel gets a column
        color = manager.colors[col]  # Pick color from Husl palette
        scatter_color = manager.scatter_colors[col] #Pick scatter color from Coolwarm palette
        for row, approx in enumerate(approximations):  # Approximation types in rows
            print("Kernel:", kernel, "Approximation:", approx)
            manager.next_subplot(row, col, title=f"{kernel_str[kernel]} - {approx}")
            K_mat_exact, K_mat_approx, x1_x2_norm = kernel_approximation(kernel=kernel, approx=approx, length_scale=2.0)
            plt.plot(x1_x2_norm, K_mat_exact, color=color, linewidth=2, label='Exact', alpha=0.5)
            plt.scatter(x1_x2_norm, K_mat_approx, color=scatter_color, marker='+', label="Approximation")


            # Optional: Show legend only in the first subplot
            if row == 0 and col == 0:
                plt.legend()
            if row == 1:
                plt.xlabel(r"$||x_1 - x_2||$", fontsize=12)
            if col == 0:
                plt.ylabel(r"$K(x_1, x_2)$", fontsize=12)
