import torch, math
import time
from torcheval.metrics.functional import r2_score
from torcheval.metrics import MeanSquaredError
from ..orthogonal_random_features import ORF_w
from ..random_fourier_features import RFF_w
from .__init__ import FormattedFileWriter
import json
from torchmetrics.classification import MulticlassCalibrationError


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = (torch.cuda.get_device_properties(DEVICE).total_memory // 1024**3 - 1)  # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # RAM available for computing
    
class RFF_ridge_regression:
    """_summary_
    Class for training of RFF EigenPro2
    Attributes:
        n_train: Number of training samples
        n_label: Number of classes/regressors
        d_dim: number of columns in the dataset
        W: Weight matrix (Kernel dependent)
        b: Bias vector
        c1: Scaling factor as per Random Fourier Features paper
        Phi_train_samp: Random features for training subsamples as per EigenPro2
        a_sol: Solution to the optimization problem
        y_train_eval: Labels for training samples to be evaluated
    """
    def __init__(self, 
        X_train,
        y_train,
        X_test,
        y_test,
        length_scale: float = 1.0,
        kernel: str = "Laplace",
        nu: float = 1.5,
        alpha: float = 0.7,
        p_feat: int = 10_000,
        Sampling_Scheme: str = "RFF",
        Rf_Bias: bool = True,
        ridge_lambda: list = [1e-2],
        n_train_eval: int = None,
        metric: str = "R2Score"):
        """_summary_
        Initializes the RFF_ridge_regression class
        Args:
            X_train (torch.tensor): Training data
            y_train (torch.tensor): Training labels
            X_test (torch.tensor): Test data
            y_test (torch.tensor): Test labels
            length_scale (float, optional): length_scale of the Kernel. Defaults to 1.0.
            kernel (str, optional): One of Laplace or Gauss or Matern. Defaults to "Laplace".
            nu (float, optional): shape parameter for Matern kernel. Defaults to 1.5.
            alpha (float, optional): shape parameter for ExpPower kernel. Defaults to 0.7.
            p_feat (int, optional): Number of random features. Defaults to 10_000.
            Sampling_Scheme (str, optional): One of RFF, ORF. Defaults to "RFF".
            Rf_Bias (bool, optional): Random features use bias or are based on SinCos formulation. Defaults to True.
            ridge_lambda (list, optional): Explicit ridge penalty. Defaults to 1e-2.
            n_train_eval (int, optional): Number of training samples to be evaluated for training metrics. Defaults to None.
            metric (str, optional): One of R2Score, MSE and Accuracy. Defaults to "R2Score".
        """
        self.X_train = X_train
        self.y_train = y_train.to(DEVICE)
        self.X_test = X_test
        self.y_test = y_test.to(DEVICE)
        del X_train, y_train, X_test, y_test
        self.length_scale = length_scale
        self.kernel_type = kernel
        self.nu = nu
        self.alpha = alpha
        self.p_feat = p_feat
        self.Sampling_Scheme = Sampling_Scheme
        self.Rf_Bias = Rf_Bias
        self.lambda_ = ridge_lambda
        self.n_train_eval = n_train_eval
        self.metric = metric
        self.d_dim = self.X_train.shape[1]
        self.n_label = self.y_train.shape[1]
        self.n_train = self.X_train.shape[0]
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        
    def set_basic_params(self):
        if self.n_train_eval is None:
            self.n_train_eval = self.n_train // 10

        self.train_eval_ids = torch.randperm(self.n_train)[ : min(self.n_train, self.n_train_eval)]

    def get_max_bs(self):
        print(f"Memory available: {DEV_MEM} GB")
        b1 = (DEV_MEM*1024**3/(8*self.p_feat))
        b2 = self.p_feat + self.d_dim + 2
        self.batch_size = int(0.25*(b1-b2))
        print(f"Batch size: {self.batch_size}")
    
    def write_attrs(self):
        attributes_ = {
        "length_scale": self.length_scale,
        "kernel": self.kernel_type,
        "p_feat": self.p_feat,
        "Sampling_Scheme": self.Sampling_Scheme,
        "Rf_Bias": self.Rf_Bias,
        "batch_size": self.batch_size,
        "n_train_eval": self.n_train_eval,
        "metric": self.metric,
        "time": time.ctime()
        }
        if self.kernel_type == "ExpPower":
            attributes_["alpha"] = self.alpha
        elif self.kernel_type == "Matern":
            attributes_["nu"] = self.nu
        
        print(f"{'Attribute':>20}  {'Value':<20}")
        print("=" * 42)
        rows = []
        row = ""
        for i, (key, value) in enumerate(attributes_.items()):
            row += f"{key:>20}  {str(value):<20}  "
            # Print the row every three items or on the last item
            if (i + 1) % 3 == 0 or i == len(attributes_) - 1:
                rows.append(row)
                row = ""
        for r in rows:
            print(r)
        self.atts = attributes_
    
    def get_W_b_c(self) -> None:
        """_summary_
        Generates W, b, c parameters depending on the kernel and scheme.
        Uses:
            self.p_feat (int): Number of random features
            self.d_dim (int): Dimension of the data
            self.kernel_type (str, optional): Defaults to "Laplace".
            self.Sampling_scheme (str, optional): Defaults to "ORF".
        """
        if not self.Rf_Bias:
            p_feat = self.p_feat // 2
        else: p_feat = self.p_feat
        
        scheme = self.Sampling_Scheme
        d = self.d_dim

        if scheme == "ORF":
            Q, R_dist = ORF_w(p_feat=p_feat, d_dim=d, Kernel=self.kernel_type, length_scale=self.length_scale, nu=self.nu, alpha=self.alpha)
            self.Q = Q.to(torch.float64).to(DEVICE)  # On GPU
            self.R_dist = R_dist.to(torch.float64).to(DEVICE)  # On GPU
            del Q, R_dist
            self.b = ((torch.rand(p_feat) * math.pi * 2).to(torch.float64).to(DEVICE))  # On GPU
            self.c1 = (torch.sqrt(torch.tensor(2 / p_feat)).to(torch.float64).to(DEVICE))  # On GPU
        elif scheme == "RFF":
            W1, W2 = RFF_w(p_feat=p_feat, d_dim=d, Kernel=self.kernel_type, length_scale=self.length_scale, nu=self.nu, alpha=self.alpha)
            self.W1 = W1.to(torch.float64).to(DEVICE)  # On GPU
            self.W2 = W2.to(torch.float64).to(DEVICE)  # On GPU
            self.b = ((torch.rand(p_feat) * math.pi * 2).to(torch.float64).to(DEVICE))  # On GPU
            self.c1 = (torch.sqrt(torch.tensor(2 / p_feat)).to(torch.float64).to(DEVICE))  # On GPU

    def create_random_features(self, X_):
        """_summary_
        Creates random features for data X_
        Args:
            X_ (_type_): Data for which random features are to be generated
            W (_type_): Weight Matrix (kernel dependent)
            b (_type_): Bias term, not used if Rf_Bias is False
            c1 (_type_): Normalization factor as per Random Fourier Features paper
            Rf_Bias (bool, optional): If random features include bias term or as SinCos features. Defaults to True.

        Returns:
            Random features created using W, b, c parameters for X_. Created on the GPU.
        """
        torch.cuda.empty_cache()
        X_ = X_.to(DEVICE)  # On GPU
        if self.Rf_Bias and self.Sampling_Scheme == 'ORF':
            return self.c1 * ((torch.mm(X_, self.Q)*self.R_dist) + self.b).cos()
        elif self.Rf_Bias and self.Sampling_Scheme == 'RFF' and self.kernel_type in ['Laplace', 'Matern']:
            return self.c1 * ((torch.mm(X_, self.W1)/self.W2) + self.b).cos()
        elif self.Rf_Bias and self.Sampling_Scheme == 'RFF' and self.kernel_type in ['Gauss', 'ExpPower']:
            return self.c1 * ((torch.mm(X_, self.W1)*self.W2) + self.b).cos()
        
        if not self.Rf_Bias and self.Sampling_Scheme == 'ORF':
            return self.c1 * torch.cat([(torch.mm(X_, self.Q)*self.R_dist).cos(), (torch.mm(X_, self.Q)*self.R_dist).sin()], dim=-1)
        elif not self.Rf_Bias and self.Sampling_Scheme == 'RFF' and self.kernel_type in ['Laplace', 'Matern']:
            return self.c1 * torch.cat([(torch.mm(X_, self.W1)/self.W2).cos(), (torch.mm(X_, self.W1)/self.W2).sin()], dim=-1)
        elif not self.Rf_Bias and self.Sampling_Scheme == 'RFF' and self.kernel_type in ['Gauss', 'ExpPower']:
            return self.c1 * torch.cat([(torch.mm(X_, self.W1)*self.W2).cos(), (torch.mm(X_, self.W1)*self.W2).sin()], dim=-1)
    
    def matrix_loader(self):
        num_rows = self.X_train.shape[0]
        for i in range(0, num_rows, self.batch_size):
            yield self.X_train[i:i+self.batch_size]
    
    def y_train_loader(self):
        num_rows = self.y_train.shape[0]
        for i in range(0, num_rows, self.batch_size):
            yield self.y_train[i:i+self.batch_size]
        
    def create_H(self):
        self.Phi_t_Phi = torch.zeros((self.p_feat, self.p_feat), dtype=torch.float64)
        i=0
        for X_batch in self.matrix_loader():
            Phi_batch = self.create_random_features(X_batch)
            self.Phi_t_Phi += (Phi_batch.T @ Phi_batch).to('cpu')
            del Phi_batch
            torch.cuda.empty_cache()
            time.sleep(1)
            
            i=i+1

    def create_Phi_t_y(self):
        self.Phi_t_ytrain = torch.zeros((self.p_feat, self.y_train.shape[1]), dtype=torch.float64)
        for X_batch, y_batch in zip(self.matrix_loader(), self.y_train_loader()):
            Phi_batch = self.create_random_features(X_batch)
            y_batch = y_batch.to(DEVICE)
            self.Phi_t_ytrain += (Phi_batch.T @ y_batch).to('cpu')
            torch.cuda.empty_cache()
    
    def calc_metrics(self):
        X_train_eval = self.X_train[self.train_eval_ids]
        self.y_train_eval = self.y_train[self.train_eval_ids]  # On GPU
        self.y_test = self.y_test.to(DEVICE)  # On GPU
        y_train_hat = self.forward(X_train_eval)  # On GPU
        y_test_hat = self.forward(self.X_test)  # On GPU
        if self.metric == "R2Score":
            self.train_score = r2_score(y_train_hat, self.y_train_eval).item()
            self.test_score = r2_score(y_test_hat, self.y_test).item()
        elif self.metric == "Accuracy":
            y_train_hat = torch.argmax(y_train_hat, dim=1)
            y_test_hat = torch.argmax(y_test_hat, dim=1)
            torch.cuda.empty_cache()
            y_train_eval = torch.argmax(self.y_train_eval, dim=1)
            y_test = torch.argmax(self.y_test, dim=1)
            self.train_score = (y_train_hat == y_train_eval).sum().item()/self.y_train_eval.shape[0]
            self.test_score = (y_test_hat == y_test).sum().item()/self.y_test.shape[0]
        elif self.metric == "Bin_Acc":
            y_sign = torch.sign(self.y_train_eval)
            y_sign_pred = torch.sign(y_train_hat)
            self.train_score = (y_sign == y_sign_pred).float().mean().item()
            y_sign = torch.sign(self.y_test)
            y_sign_pred = torch.sign(y_test_hat)
            self.test_score = (y_sign == y_sign_pred).float().mean().item()
        elif self.metric == "MSE":
            metric = MeanSquaredError(device=DEVICE)
            metric.update(y_train_hat, self.y_train_eval)
            self.train_score = metric.compute().item()
            metric.update(y_test_hat, self.y_test)
            self.test_score = metric.compute().item()
            del metric
        elif self.metric == "ECE":
            mcce = MulticlassCalibrationError(num_classes=10, n_bins=15, norm='l2')
            y_train_eval = torch.argmax(self.y_train_eval, dim=1)
            y_test = torch.argmax(self.y_test, dim=1)
            y_train_hat_softmax = torch.softmax(y_train_hat, dim = 1)
            y_test_hat_softmax = torch.softmax(y_test_hat, dim = 1)
            self.train_score = mcce(y_train_hat_softmax, y_train_eval).item()
            self.test_score = mcce(y_test_hat_softmax, y_test).item()
        del y_train_hat, y_test_hat, X_train_eval
        torch.cuda.empty_cache()

    def forward(self, X_):
        """_summary_
        Does Phi(X) @ a_sol in a way where memory overflow does not occur
        Args:
            X_ (torch.tensor): Value for which prediction is to be made
        Returns:
            y_hat: Prediction corresponding to X_, stored on GPU
        """
        n_samples = X_.shape[0]
        batches = torch.randperm(n_samples).split(self.batch_size)
        y_hat = torch.zeros(n_samples, self.n_label, device=DEVICE) # On GPU
        for _, b_ids in enumerate(batches):
            self.PHI = self.create_random_features(X_[b_ids])  # On GPU
            y_hat[b_ids] = self.PHI @ self.a_sol  # On GPU
            # print(colored(f"Current allocated memory {torch.cuda.memory_allocated()/(1024**3):,.4f}", "light_red"),"GB")
            del self.PHI
            torch.cuda.empty_cache()
        return y_hat  # On GPU

    def fit(self):
        self.start_time = time.time()
        self.set_basic_params()
        self.get_max_bs()
        self.write_attrs()
        self.get_W_b_c()
        self.create_H()
        print("H computed")
        self.create_Phi_t_y()
        self.Phi_t_ytrain = self.Phi_t_ytrain.to(DEVICE)
        stop_time = time.time()
        print(f"Matrix assembly time: {stop_time - self.start_time:.2f}")
        metric_dict ={"train":{},
                      "test":{}}
        for r_lam in self.lambda_:
            r_lam_str = f'{r_lam:.0e}'
            inv_start_time = time.time()
            self.Phi_t_Phi_lam = (self.Phi_t_Phi + r_lam * torch.eye(self.p_feat)).to(DEVICE)
            self.a_sol = torch.linalg.solve(self.Phi_t_Phi_lam, self.Phi_t_ytrain)
            print(f"Inversion time: {time.time() - inv_start_time:.2f}")
            del self.Phi_t_Phi_lam
            torch.cuda.empty_cache()
            self.calc_metrics()
            print(f"lambda: {r_lam:.4E} \t train_score: {self.train_score:.4f} \t test_score: {self.test_score:.4f}")
            metric_dict["train"][r_lam_str] = self.train_score
            metric_dict["test"][r_lam_str] = self.test_score
        return metric_dict


n = 20_000  # number of samples
d = 8  # dimensions
c = 1  # number of targets
w_star = torch.rand(d, c)
normalize = lambda x: x / x.norm(dim=-1, keepdim=True)
x_train, x_test = normalize(torch.randn(n, d)), normalize(torch.randn(n // 2, d))
y_train, y_test = torch.sign(x_train @ w_star), torch.sign(x_test @ w_star)
p = 1000

model = RFF_ridge_regression(x_train, y_train, x_test, y_test, p_feat=p, Rf_Bias=True, ridge_lambda=[1e-2, 1e-3, 1e-5, 1e-8],    
                                kernel="ExpPower", metric="Bin_Acc", alpha=1.2)
error_dict = model.fit()
print("Ridge Regression test is complete")