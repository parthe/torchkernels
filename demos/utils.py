import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torchkernels.kernels.radial import laplacian, gaussian, exp_power, matern

def load_fmnist(device='cpu'):
	transform = transforms.Compose([transforms.ToTensor()])
	root_dir = os.environ.get("DATASETS_DIR", "./data")
	train_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=True, transform=transform, download=True)
	test_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=False, transform=transform, download=True)

	X_train = train_dataset.data.to(device).flatten(1).float()/255.0
	X_test = test_dataset.data.to(device).flatten(1).float()/255.0
	normalize = lambda x : x/x.norm(dim=-1, keepdim=True)
	X_train, X_test = normalize(X_train), normalize(X_test)
	y_train = train_dataset.targets.to(device)
	y_test = test_dataset.targets.to(device)

	return X_train, X_test, y_train, y_test

plt.rcParams.update({
		"text.usetex": False,
		"font.family": "serif",
		"mathtext.fontset": "cm",
		"mathtext.rm": "serif",
	})


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
		# self._add_vertical_lines()  # Add lines before showing
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


def create_kernel_str(kernel:str, nu:float = 1.5, alpha:float = 0.7):
	assert kernel in ["Laplacian", "Gaussian", "Matern", "ExpPower"]
	if kernel == "ExpPower":
		return r"ExpPower ($\alpha = %.2f$)" % alpha
	elif kernel == "Matern":
		return r"Matérn ($\nu = %.2f$)" % nu
	else:
		return kernel
	

def create_x1_x2_list(input_dim:int, N:int=20, start:float=0, end:float=5):
	x1_x2_norm = np.linspace(start, end, N)
	x1_list = []
	x2_list = []
	for i, x_ in enumerate(x1_x2_norm):
		x1 = torch.randn(input_dim)
		x1_x2_u = torch.randn(input_dim)
		x1_x2 = x1_x2_u/np.linalg.norm(x1_x2_u)*x_
		x2 = x1 + x1_x2
		x1 = x1.reshape(1,-1)
		x2 = x2.reshape(1,-1)
		x1_list.append(x1)
		x2_list.append(x2)
	return x1_list, x2_list, x1_x2_norm


def create_Kmat_exact(kernels:list, x1_list:list, x2_list:list, length_scale:float=1., nu:float=1.5, alpha:float=0.7):
	K_func_exact = {"Laplacian": laplacian, "Gaussian": gaussian, "Matern": matern, "ExpPower": exp_power}
	K_mat_dict = {}
	for kernel in kernels:
		K_mat_dict[kernel] = []
		for x1,x2 in zip(x1_list, x2_list):
			if kernel == "ExpPower":
				K_mat_dict[kernel].append(K_func_exact[kernel](x1, x2, length_scale=length_scale, alpha=alpha)[0].item())
			elif kernel == "Matern":
				K_mat_dict[kernel].append(K_func_exact[kernel](x1, x2, length_scale=length_scale, nu=nu)[0].item())
			else:
				K_mat_dict[kernel].append(K_func_exact[kernel](x1, x2, length_scale=length_scale)[0].item())
	return K_mat_dict


def create_plots(kernels, approximations, K_exact_dict:dict, K_approx_dict:dict, x1_x2_norm:np.ndarray, kernel_str_dict:str):
	with SubplotManager(rows=2, cols=4, kernels=kernels) as manager:
		for col, kernel in enumerate(kernels):  # Each kernel gets a column
			color = manager.colors[col]  # Pick color from Husl palette
			scatter_color = manager.scatter_colors[col] #Pick scatter color from Coolwarm palette
			for row, approx in enumerate(approximations):  # Approximation types in rows
				manager.next_subplot(row, col, title=f"{kernel_str_dict[kernel]} - {approx}")
				K_mat_exact = K_exact_dict[kernel]
				K_mat_approx = K_approx_dict[kernel][approx]
				plt.plot(x1_x2_norm, K_mat_exact, color=color, linewidth=2, label='Exact', alpha=0.5)
				plt.scatter(x1_x2_norm, K_mat_approx, color=scatter_color, marker='+', label="Approximation")
				# Optional: Show legend only in the first subplot
				if row == 0 and col == 0:
					plt.legend()
				if row == 1:
					plt.xlabel(r"$||x_1 - x_2||$", fontsize=12)
				if col == 0:
					plt.ylabel(r"$K(x_1, x_2)$", fontsize=12)


def plot_kernel_approximation(x, y1, y2, kernel_str):
	"""
	Plots a single kernel approximation with line and scatter plots.

	Parameters:
		x (list or array): X-axis values.
		y1 (list or array): Y-axis values for the first scatter plot.
		y2 (list or array): Y-axis values for the second scatter plot.
	"""

	# Apply LaTeX-like styling without full LaTeX rendering
	plt.rcParams.update({
		"text.usetex": False,
		"font.family": "serif",
		"mathtext.fontset": "cm",
		"mathtext.rm": "serif",
	})
	plt.rcParams['font.family'] = "serif"

	# Set Seaborn style
	sns.set_style("whitegrid")

	# Define colors
	line_color = sns.color_palette("husl", 1)[0]
	scatter_color = sns.color_palette("icefire", 1)[0]

	# Plot line and scatter points
	plt.plot(x, y1, color=line_color, linewidth=2, label="Exact", alpha=0.5)
	plt.scatter(x, y1, color=scatter_color, marker="+", label="Approximation", s=50)
	plt.xlabel(r"$||x_1 - x_2||$", fontsize=12)
	plt.ylabel(r"$K(x_1, x_2)$", fontsize=12)

	# Set title and grid
	plt.title(f"{kernel_str}", fontname="serif",  fontsize=12)
	plt.grid(True)
	import matplotlib.font_manager as font_manager
	font = font_manager.FontProperties(family='serif', style='normal', size=10)
	# Add legend
	plt.legend(prop=font)

	# Show plot
	plt.show()

