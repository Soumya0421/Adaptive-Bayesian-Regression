"""
Auto Bayesian Polynomial Regression (ABPR)

A automic Bayesian Polynomial Regression algorithm that requires no manual hyperparameters tunning.

The algorithm combines:
    - Orthogonal Distance Regression
    - Student's t error model
    - Bayesian Automatic Relevance Determination (ARD)
    - Bayesian model evidence for automatic polynomial degree selection

The algorithm is written only using Numpy and SciPy.
"""

import os
import sys
import pickle
import logging
import itertools

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats
from scipy import optimize

# Set seed
np.random.seed(42)

class AutoBayesianPolynomialRegression:
    """
    Automatic Bayesian Polynomial Regression Model.
    """

    def __init__(self, config: dict | None = None) -> None:
        """
        Parameters:
        ----------
        max_degree : int
            Maximum polynomial degree.
        patience : int
            Number of consecutive non-improving degrees iteration before stopping.
        evidence_tol : float
            Minimum evidence improvement to count as "improved".
        max_iter : int
            Maximum iterations per degree.
        tol : float
            Convergence tolerance on the evidence within a degree fit.
        latent_lr : float
            Step size for the latent-input (ODR) gradient updates.
        verbose : bool
            Whether to emit [INFO] log statements.
        """
        config = config or {}
        self.max_degree = config.get("max_degree", 10)
        self.patience = config.get("patience", 3)
        self.evidence_tol = config.get("evidence_tol", 1e-2)
        self.max_iter = config.get("max_iter", 200)
        self.tol = config.get("tol", 1e-3)
        self.latent_lr = config.get("latent_lr", 0.05)
        self.verbose = config.get("verbose", True)

        # Initialize logger
        self.logger = logging.getLogger("ABPR")
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        self.logger.propagate = False

        # Fitted attributes
        self.mu_ = None
        self.sigma_ = None
        self.mu_y_ = None
        self.sigma_y_ = None
        self.degree_ = None
        self.exponents_ = None
        self.beta_ = None
        self.alpha_ = None
        self.sigma2_ = None
        self.nu_ = None
        self.S_ = None
        self.evidence_ = None
        self.history_ = None

    # Logging
    def _log(self, msg):
        self.logger.info(msg)

    # Data preprocessing
    def preprocess(self, filepath, target_col):
        self._log(f"[INFO] Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        if target_col and target_col in df.columns:
            y = df[target_col].values
            X = df.drop(columns=[target_col]).values
        else:
            y = None
            X = df.values          
        return X, y

    # Standardization
    def _standardize_fit(self, X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0.0, 1.0, sigma)
        return mu, sigma

    def _standardize_transform(self, X, mu, sigma):
        return (X - mu) / sigma

    # Polynomial feature expansion
    def _generate_exponents(self, p, degree):
        exponents = []
        for total_degree in range(degree + 1):
            for combo in itertools.combinations_with_replacement(range(p), total_degree):
                exp = [0] * p
                for idx in combo:
                    exp[idx] += 1
                exponents.append(exp)
        return np.array(exponents, dtype=float)

    def _poly_features(self, Z, exponents):
        N = Z.shape[0]
        M = exponents.shape[0]
        Phi = np.ones((N, M))
        for m in range(M):
            exp = exponents[m]
            for k in range(Z.shape[1]):
                if exp[k] != 0:
                    Phi[:, m] = Phi[:, m] * (Z[:, k] ** exp[k])
        return Phi

    def _poly_jacobian(self, Z, exponents):
        N, p = Z.shape
        M = exponents.shape[0]
        J = np.zeros((N, M, p))
        for m in range(M):
            exp = exponents[m]
            for k in range(p):
                if exp[k] == 0:
                    continue
                term = exp[k] * (Z[:, k] ** (exp[k] - 1))
                for l in range(p):
                    if l == k or exp[l] == 0:
                        continue
                    term = term * (Z[:, l] ** exp[l])
                J[:, m, k] = term
        return J

    # Robust (Student's t) degrees-of-freedom update
    def _update_nu(self, residuals, scale2, nu_bounds=(2.0, 200.0)):
        scale = np.sqrt(max(float(scale2), 1e-10))

        def neg_log_likelihood(log_nu):
            nu = np.exp(log_nu)
            return -np.sum(stats.t.logpdf(residuals, df=nu, scale=scale))

        result = optimize.minimize_scalar(
            neg_log_likelihood,
            bounds=(np.log(nu_bounds[0]), np.log(nu_bounds[1])),
            method="bounded",
        )
        return float(np.exp(result.x))

    def _logdet(self, S):
        try:
            L = linalg.cholesky(S, lower=True)
            return 2.0 * np.sum(np.log(np.clip(np.diag(L), 1e-300, None)))
        except linalg.LinAlgError:
            eigvals = linalg.eigvalsh(S)
            eigvals = eigvals[eigvals > 1e-12]
            if eigvals.size == 0:
                return None
            return float(np.sum(np.log(eigvals)))

    # Bayesian evidence (Laplace / MacKay-style approximation)
    def _log_evidence(self, Phi, y, beta, alpha, sigma2, nu, S, Z, X, nu_x, sigma_x2):
        resid = y - Phi @ beta
        scale = np.sqrt(max(sigma2, 1e-10))
        log_likelihood = np.sum(stats.t.logpdf(resid, df=nu, scale=scale))

        logdet_S = self._logdet(S)
        complexity_term = 0.5 * logdet_S if logdet_S is not None else -50.0

        prior_term = 0.5 * np.sum(np.log(np.clip(alpha, 1e-12, None)))

        scale_x = np.sqrt(np.maximum(sigma_x2, 1e-10))
        log_px_z = np.sum(stats.t.logpdf(Z - X, df=nu_x, scale=scale_x[None, :]))

        return log_likelihood + prior_term + complexity_term + log_px_z

    # Core Bayesian fit for one fixed polynomial degree
    def _fit_single_degree(self, X, y, degree):
        N, p = X.shape
        exponents = self._generate_exponents(p, degree)
        M = exponents.shape[0]

        self._log(f"[INFO] Generated {M} polynomial features")
        self._log("[INFO] Initializing Bayesian parameters")

        # Latent (error-corrected) inputs, initialized at the observed values
        Z = X.copy()
        sigma_x2 = np.var(X, axis=0) + 1e-6
        nu_x = 5.0

        # Regression / noise / ARD parameters
        alpha = np.ones(M)
        sigma2 = float(np.var(y) + 1e-6)
        nu = 5.0
        beta = np.zeros(M)
        S = np.eye(M)

        prev_evidence = -np.inf
        evidence = -np.inf

        for iteration in range(self.max_iter):
            Phi = self._poly_features(Z, exponents)

            # E-step: Student's t robustness weights on residuals 
            resid = y - Phi @ beta
            w = (nu + 1.0) / (nu + (resid ** 2) / sigma2)

            # M-step: weighted Bayesian ridge regression for beta 
            self._log("[INFO] Updating regression coefficients...")
            WPhi = Phi * (w / sigma2)[:, None]
            jitter = 1e-8 * np.trace(Phi.T @ WPhi) / M if M > 0 else 1e-4
            precision = Phi.T @ WPhi + np.diag(alpha) + jitter * np.eye(M)
            try:
                S = linalg.inv(precision)
            except linalg.LinAlgError:
                S = linalg.pinv(precision)
            beta = S @ (Phi.T @ (w * y / sigma2))

            # ARD precision update (damped to avoid runaway shrinkage) 
            self._log("[INFO] Updating ARD precisions...")
            diag_S = np.clip(np.diag(S), 1e-12, None)
            gamma = np.clip(1.0 - alpha * diag_S, 1e-8, None)
            alpha_candidate = np.clip(gamma / (beta ** 2 + 1e-6), 1e-8, 1e6)
            alpha = 0.5 * alpha + 0.5 * alpha_candidate

            # Noise scale update
            resid = y - Phi @ beta
            w = (nu + 1.0) / (nu + (resid ** 2) / sigma2)
            eff_N = max(N - float(np.sum(gamma)), 1.0)
            sigma2 = max(float(np.sum(w * resid ** 2) / eff_N), 1e-8)

            # Student's t degrees-of-freedom update
            self._log("[INFO] Updating Student's t parameters...")
            nu = self._update_nu(resid, sigma2)

            # Latent input update (Orthogonal Distance Regression)
            self._log("[INFO] Updating latent inputs...")
            J = self._poly_jacobian(Z, exponents)                
            grad_pred = np.einsum("nmp,m->np", J, beta) 
            input_err = Z - X
            wx = (nu_x + 1.0) / (nu_x + (input_err ** 2) / sigma_x2[None, :])

            grad = (-(w * resid / sigma2)[:, None] * grad_pred + wx * input_err / sigma_x2[None, :])

            # Safety clip: guards against transient blow-ups early in training
            grad = np.clip(grad, -50.0, 50.0)
            Z = Z - self.latent_lr * grad

            # Floor sigma_x2 relative to the (standardized) feature scale rather than at an absolute epsilon
            sigma_x2_floor = 1e-3 * np.maximum(np.var(X, axis=0), 1e-6)
            sigma_x2 = np.maximum(np.sum(wx * input_err ** 2, axis=0) / N, sigma_x2_floor)
            nu_x = self._update_nu(input_err.ravel(), float(np.mean(sigma_x2)))

            # Evidence
            self._log("[INFO] Computing Bayesian evidence...")
            evidence = self._log_evidence(Phi, y, beta, alpha, sigma2, nu, S, Z, X, nu_x, sigma_x2)

            if abs(evidence - prev_evidence) < self.tol:
                break
            prev_evidence = evidence

        return {
            "exponents": exponents,
            "beta": beta,
            "alpha": alpha,
            "sigma2": sigma2,
            "nu": nu,
            "S": S,
            "Z": Z,
            "sigma_x2": sigma_x2,
            "nu_x": nu_x,
            "evidence": evidence,
        }

    # Automatic polynomial degree search (driven by Bayesian evidence)
    def _search_degree(self, X, y):
        best_evidence = -np.inf
        best_model = None
        patience_counter = 0
        degree = 1
        history = []

        while True:
            self._log(f"[INFO] Polynomial degree = {degree}")
            model = self._fit_single_degree(X, y, degree)
            evidence = model["evidence"]
            history.append({"degree": degree, "evidence": evidence})

            if evidence > best_evidence + self.evidence_tol:
                self._log("[INFO] Evidence improved")
                best_evidence = evidence
                best_model = dict(model)
                best_model["degree"] = degree
                patience_counter = 0
                self._log(f"[INFO] Best degree updated to {degree}")
            else:
                patience_counter += 1
                self._log(f"[INFO] Patience = {patience_counter}/{self.patience}")

            if patience_counter >= self.patience or degree >= self.max_degree:
                break
            degree += 1

        return best_model, history

    # Public API
    def fit(self, filepath, target_col):
        X, y = self.preprocess(filepath, target_col)
        if y is None:
            raise ValueError(f"Target column '{target_col}' not found in {filepath}.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._log("[INFO] Standardizing features and target...")
        self.mu_, self.sigma_ = self._standardize_fit(X)
        Z = self._standardize_transform(X, self.mu_, self.sigma_)

        self.mu_y_ = float(np.mean(y))
        self.sigma_y_ = float(np.std(y))
        if self.sigma_y_ == 0.0:
            self.sigma_y_ = 1.0
        y_scaled = (y - self.mu_y_) / self.sigma_y_

        best_model, history = self._search_degree(Z, y_scaled)

        if best_model is None:
            raise RuntimeError("No valid model found during degree search. Check input data or parameters.")

        self.degree_ = best_model["degree"]
        self.exponents_ = best_model["exponents"]
        self.beta_ = best_model["beta"]
        self.alpha_ = best_model["alpha"]
        self.sigma2_ = best_model["sigma2"]
        self.nu_ = best_model["nu"]
        self.S_ = best_model["S"]
        self.evidence_ = best_model["evidence"]
        self.history_ = history if history is not None else []

        self._log("[INFO] Training completed.")
        return self

    def predict(self, filepath, target_col=None, return_std=False):
        if self.beta_ is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

        X, _ = self.preprocess(filepath, target_col)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        Z = self._standardize_transform(X_arr, self.mu_, self.sigma_)
        Phi = self._poly_features(Z, self.exponents_)
        y_pred_scaled = Phi @ self.beta_
        y_pred = (y_pred_scaled * self.sigma_y_) + self.mu_y_
        std = None

        if return_std:
            sigma2 = float(self.sigma2_) if self.sigma2_ is not None else 0.0
            if self.S_ is not None:
                var = sigma2 + np.einsum("nm,mk,nk->n", Phi, self.S_, Phi)
            else:
                var = sigma2 * np.ones(len(y_pred_scaled))
            std = np.sqrt(np.clip(var, 1e-12, None)) * self.sigma_y_

        # Read the original file to keep feature column names intact
        original_df = pd.read_csv(filepath)
        if target_col and target_col in original_df.columns:
            output_df = original_df.drop(columns=[target_col]).copy()
        else:
            output_df = original_df.copy()
            
        output_df["Predicted"] = y_pred
        
        if return_std:
            output_df["Predicted_Std"] = std
            
        # Safely create output directory
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        output_filename = "predicted.csv"
        output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename
        
        output_df.to_csv(output_path, index=False)
        self._log(f"[INFO] Saved predictions to {output_path}")

        if not return_std:
            return y_pred
            
        return y_pred, std

    def summary(self):
        if self.beta_ is None:
            return "Model has not been fit yet."
        n_active = int(np.sum(np.abs(self.beta_) > 1e-4))
        return (
            f"ABPR fitted model\n"
            f"  degree            : {self.degree_}\n"
            f"  total terms       : {len(self.beta_)}\n"
            f"  active terms      : {n_active}\n"
            f"  noise variance    : {self.sigma2_:.6g}\n"
            f"  residual dof (nu) : {self.nu_:.3f}\n"
            f"  log evidence      : {self.evidence_:.4f}"
        )
        
    # Persistence
    def save(self, filepath):
        if self.beta_ is None:
            raise RuntimeError("Cannot save an unfitted model. Call .fit() first.")

        state = {
            "hyperparams": {
                "max_degree": self.max_degree,
                "patience": self.patience,
                "evidence_tol": self.evidence_tol,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "latent_lr": self.latent_lr,
                "verbose": self.verbose,
            },
            "fitted": {
                "mu_": self.mu_,
                "sigma_": self.sigma_,
                "mu_y_": self.mu_y_,
                "sigma_y_": self.sigma_y_,
                "degree_": self.degree_,
                "exponents_": self.exponents_,
                "beta_": self.beta_,
                "alpha_": self.alpha_,
                "sigma2_": self.sigma2_,
                "nu_": self.nu_,
                "S_": self.S_,
                "evidence_": self.evidence_,
                "history_": self.history_,
            },
        }

        # Safely create output directory
        save_dir = os.path.dirname(filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._log(f"[INFO] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            state = f.read()
        state = pickle.loads(state)

        model = cls(**state["hyperparams"])
        for key, value in state["fitted"].items():
            setattr(model, key, value)

        model._log(f"[INFO] Model loaded from {filepath}")
        return model