"""
Auto Bayesian Polynomial Regression (ABPR)

A automic Bayesian Polynomial Regression algorithm that requires no manual hyperparameters tunning.

The algorithm combines:
    - Orthogonal Distance Regression
    - Student's t error model
    - Bayesian Automatic Relevance Determination (ARD)
    - Bayesian model evidence for automatic polynomial degree selection
    - Native multi-output (multi-target) regression

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
        self.target_names_ = None
        self.n_targets_ = None
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

        if not target_col:
            return df.values, None, None

        target_cols = list(target_col) if isinstance(target_col, (list, tuple)) else [target_col]
        present = [c for c in target_cols if c in df.columns]

        if len(present) != len(target_cols):
            return df.values, None, target_cols

        y = df[target_cols].values.astype(float)
        X = df.drop(columns=target_cols).values
        return X, y, target_cols

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
        K = y.shape[1]
        total = 0.0
        for k in range(K):
            resid_k = y[:, k] - Phi @ beta[:, k]
            scale_k = np.sqrt(max(sigma2[k], 1e-10))
            log_likelihood_k = np.sum(stats.t.logpdf(resid_k, df=nu[k], scale=scale_k))

            logdet_Sk = self._logdet(S[k])
            complexity_k = 0.5 * logdet_Sk if logdet_Sk is not None else -50.0

            prior_k = 0.5 * np.sum(np.log(np.clip(alpha[:, k], 1e-12, None)))

            total += log_likelihood_k + complexity_k + prior_k

        scale_x = np.sqrt(np.maximum(sigma_x2, 1e-10))
        log_px_z = np.sum(stats.t.logpdf(Z - X, df=nu_x, scale=scale_x[None, :]))

        return total + log_px_z

    # Core Bayesian fit 
    def _fit_single_degree(self, X, y, degree):
        N, p = X.shape
        K = y.shape[1]
        exponents = self._generate_exponents(p, degree)
        M = exponents.shape[0]

        self._log(f"[INFO] Generated {M} polynomial features for {K} target(s)")
        self._log("[INFO] Initializing Bayesian parameters")

        # Latent (error-corrected) inputs, initialized at the observed values.
        # Shared across all outputs, since X is shared.
        Z = X.copy()
        sigma_x2 = np.var(X, axis=0) + 1e-6
        nu_x = 5.0

        # Regression / noise / ARD parameters, one column/entry per output
        alpha = np.ones((M, K))
        sigma2 = np.array([np.var(y[:, k]) + 1e-6 for k in range(K)])
        nu = np.full(K, 5.0)
        beta = np.zeros((M, K))
        S = np.array([np.eye(M) for _ in range(K)])

        prev_evidence = -np.inf
        evidence = -np.inf

        for iteration in range(self.max_iter):
            self._log("")
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}/{self.max_iter}")

            Phi = self._poly_features(Z, exponents)

            # E-step: Student's t robustness weights on residuals (per output)
            resid = y - Phi @ beta
            w = (nu[None, :] + 1.0) / (nu[None, :] + (resid ** 2) / sigma2[None, :])

            # M-step: weighted Bayesian ridge regression for beta, solved independently, per output
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: Updating regression coefficients...")
            new_beta = np.zeros((M, K))
            new_S = np.zeros((K, M, M))
            for k in range(K):
                WPhi = Phi * (w[:, k] / sigma2[k])[:, None]
                jitter = 1e-8 * np.trace(Phi.T @ WPhi) / M if M > 0 else 1e-4
                precision = Phi.T @ WPhi + np.diag(alpha[:, k]) + jitter * np.eye(M)
                try:
                    Sk = linalg.inv(precision)
                except linalg.LinAlgError:
                    Sk = linalg.pinv(precision)
                new_S[k] = Sk
                new_beta[:, k] = Sk @ (Phi.T @ (w[:, k] * y[:, k] / sigma2[k]))
            S = new_S
            beta = new_beta

            # ARD precision update (damped to avoid runaway shrinkage), per output
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: Updating ARD precisions...")
            gamma = np.zeros((M, K))
            for k in range(K):
                diag_Sk = np.clip(np.diag(S[k]), 1e-12, None)
                gamma_k = np.clip(1.0 - alpha[:, k] * diag_Sk, 1e-8, None)
                alpha_candidate_k = np.clip(gamma_k / (beta[:, k] ** 2 + 1e-6), 1e-8, 1e6)
                alpha[:, k] = 0.5 * alpha[:, k] + 0.5 * alpha_candidate_k
                gamma[:, k] = gamma_k

            # Noise scale update, per output
            resid = y - Phi @ beta
            w = (nu[None, :] + 1.0) / (nu[None, :] + (resid ** 2) / sigma2[None, :])
            for k in range(K):
                eff_N_k = max(N - float(np.sum(gamma[:, k])), 1.0)
                sigma2[k] = max(float(np.sum(w[:, k] * resid[:, k] ** 2) / eff_N_k), 1e-8)

            # Student's t degrees-of-freedom update, per output
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: Updating Student's t parameters...")
            for k in range(K):
                nu[k] = self._update_nu(resid[:, k], sigma2[k])

            # Latent input update (Orthogonal Distance Regression)
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: Updating latent inputs (ODR)...")
            J = self._poly_jacobian(Z, exponents)
            input_err = Z - X
            wx = (nu_x + 1.0) / (nu_x + (input_err ** 2) / sigma_x2[None, :])

            grad = np.zeros((N, p))
            for k in range(K):
                grad_pred_k = np.einsum("nmp,m->np", J, beta[:, k])
                grad += -(w[:, k] * resid[:, k] / sigma2[k])[:, None] * grad_pred_k
            grad += wx * input_err / sigma_x2[None, :]

            # Safety clip: guards against transient blow-ups early in training
            grad = np.clip(grad, -50.0, 50.0)
            Z = Z - self.latent_lr * grad

            # Floor sigma_x2 relative to the (standardized) feature scale rather than at an absolute epsilon
            sigma_x2_floor = 1e-3 * np.maximum(np.var(X, axis=0), 1e-6)
            sigma_x2 = np.maximum(np.sum(wx * input_err ** 2, axis=0) / N, sigma_x2_floor)
            nu_x = self._update_nu(input_err.ravel(), float(np.mean(sigma_x2)))

            # Evidence
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: Computing Bayesian evidence...")
            evidence = self._log_evidence(Phi, y, beta, alpha, sigma2, nu, S, Z, X, nu_x, sigma_x2)
            self._log(f"[INFO] Degree {degree} | Iteration {iteration + 1}: log evidence = {evidence:.6f}")

            if abs(evidence - prev_evidence) < self.tol:
                self._log(f"[INFO] Degree {degree}: converged after {iteration + 1} iteration(s) "
                          f"(|delta evidence| < {self.tol})")
                break
            prev_evidence = evidence
        else:
            self._log(f"[INFO] Degree {degree}: reached max_iter ({self.max_iter}) without convergence")

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
            self._log("")
            self._log("")
            self._log(f"[INFO] Starting fit for polynomial degree = {degree} (max_degree = {self.max_degree})")
            model = self._fit_single_degree(X, y, degree)
            evidence = model["evidence"]
            history.append({"degree": degree, "evidence": evidence})

            self._log("")
            self._log(f"[INFO] Degree {degree} finished | log evidence = {evidence:.6f}")

            if evidence > best_evidence + self.evidence_tol:
                self._log(f"[INFO] Degree {degree}: evidence improved over previous best "
                          f"({best_evidence:.6f} -> {evidence:.6f})")
                best_evidence = evidence
                best_model = dict(model)
                best_model["degree"] = degree
                patience_counter = 0
                self._log(f"[INFO] Best degree updated to {degree}")
            else:
                patience_counter += 1
                self._log(f"[INFO] Degree {degree}: no significant evidence improvement "
                          f"(patience {patience_counter}/{self.patience})")

            if patience_counter >= self.patience:
                self._log(f"[INFO] Stopping degree search: patience limit reached at degree {degree}")
                break
            if degree >= self.max_degree:
                self._log(f"[INFO] Stopping degree search: reached max_degree = {self.max_degree}")
                break
            degree += 1

        return best_model, history

    # Public API
    def fit(self, filepath, target_col):
        X, y, target_cols = self.preprocess(filepath, target_col)
        if y is None:
            raise ValueError(f"Target column(s) '{target_col}' not found in {filepath}.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.target_names_ = target_cols
        if self.target_names_ is None:
            raise RuntimeError("Target column names could not be determined after preprocessing.")
        self.n_targets_ = y.shape[1]
        self._log(f"[INFO] Detected {self.n_targets_} target column(s): {', '.join(self.target_names_)}")

        self._log("[INFO] Standardizing features and target(s)...")
        self.mu_, self.sigma_ = self._standardize_fit(X)
        Z = self._standardize_transform(X, self.mu_, self.sigma_)

        self.mu_y_ = y.mean(axis=0)
        self.sigma_y_ = y.std(axis=0)
        self.sigma_y_ = np.where(self.sigma_y_ == 0.0, 1.0, self.sigma_y_)
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

        X, _, _ = self.preprocess(filepath, target_col)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.mu_y_ is None or self.sigma_y_ is None:
            raise RuntimeError("Model target scaling parameters are missing. Refit the model or load a valid trained model.")

        Z = self._standardize_transform(X_arr, self.mu_, self.sigma_)
        Phi = self._poly_features(Z, self.exponents_)
        y_pred_scaled = Phi @ self.beta_                       # (N, K)

        sigma_y = np.atleast_1d(np.asarray(self.sigma_y_, dtype=float))
        mu_y = np.atleast_1d(np.asarray(self.mu_y_, dtype=float))
        if sigma_y.ndim == 0:
            sigma_y = sigma_y.reshape(1)
        if mu_y.ndim == 0:
            mu_y = mu_y.reshape(1)

        y_pred = (y_pred_scaled * sigma_y[None, :]) + mu_y[None, :]

        if self.n_targets_ is None:
            raise RuntimeError("Number of targets is missing. Refit the model or load a valid trained model.")
        K = int(self.n_targets_)
        std = None
        if return_std:
            if self.sigma2_ is None or self.S_ is None or self.sigma_y_ is None:
                raise RuntimeError("Model uncertainty parameters are missing. Refit the model or load a valid trained model.")
            num_samples = int(y_pred.shape[0])
            std = np.zeros((num_samples, K))
            for k in range(K):
                var_k = self.sigma2_[k] + np.einsum("nm,mk,nk->n", Phi, self.S_[k], Phi)
                std[:, k] = np.sqrt(np.clip(var_k, 1e-12, None)) * self.sigma_y_[k]
            if K == 1:
                std = std.reshape(-1, 1)

        # Read the original file to keep feature column names intact
        original_df = pd.read_csv(filepath)
        target_cols = self.target_names_ if self.target_names_ is not None else None
        # Ensure target_cols is iterable; if None, create placeholder names for multi-targets
        if target_cols is None:
            if K == 1:
                target_cols = []
            else:
                target_cols = [f"Target_{i}" for i in range(K)]

        drop_cols = [c for c in target_cols if c in original_df.columns]
        output_df = original_df.drop(columns=drop_cols).copy() if drop_cols else original_df.copy()

        if K == 1:
            output_df["Predicted"] = y_pred[:, 0]
            if return_std:
                if std is None:
                    raise RuntimeError("Standard deviations requested but could not be computed.")
                output_df["Predicted_Std"] = std[:, 0]
        else:
            if return_std:
                if std is None:
                    raise RuntimeError("Standard deviations requested but could not be computed.")
                for k, name in enumerate(target_cols):
                    output_df[f"Predicted_{name}"] = y_pred[:, k]
                    output_df[f"Predicted_Std_{name}"] = std[:, k]
            else:
                for k, name in enumerate(target_cols):
                    output_df[f"Predicted_{name}"] = y_pred[:, k]

        # Safely create output directory
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_filename = "predicted.csv"
        output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename

        output_df.to_csv(output_path, index=False)
        self._log(f"[INFO] Saved predictions to {output_path}")

        y_pred_out = y_pred[:, 0] if K == 1 else y_pred
        if not return_std:
            return y_pred_out

        if std is None:
            raise RuntimeError("Standard deviations requested but could not be computed.")
        std_out = std[:, 0] if K == 1 else std
        return y_pred_out, std_out

    def summary(self):
        if self.beta_ is None:
            return "Model has not been fit yet."

        lines = []
        lines.append("ABPR fitted model")
        lines.append(f"  degree            : {self.degree_}")
        lines.append(f"  total terms       : {self.beta_.shape[0]}")
        target_names = self.target_names_ if self.target_names_ is not None else [f"Target_{i}" for i in range(self.n_targets_ or 0)]
        lines.append(f"  targets           : {', '.join(target_names) if target_names else 'N/A'}")
        lines.append(f"  log evidence      : {self.evidence_:.4f}")

        sigma2 = None
        nu = None
        if self.sigma2_ is not None:
            sigma2 = np.atleast_1d(np.asarray(self.sigma2_, dtype=float))
        if self.nu_ is not None:
            nu = np.atleast_1d(np.asarray(self.nu_, dtype=float))

        for k, name in enumerate(target_names):
            n_active = int(np.sum(np.abs(self.beta_[:, k]) > 1e-4))
            lines.append("")
            lines.append(f"  [{name}]")
            lines.append(f"    active terms      : {n_active}")
            if sigma2 is not None and k < sigma2.shape[0]:
                lines.append(f"    noise variance    : {sigma2[k]:.6g}")
            else:
                lines.append("    noise variance    : N/A")
            if nu is not None and k < nu.shape[0]:
                lines.append(f"    residual dof (nu) : {nu[k]:.3f}")
            else:
                lines.append("    residual dof (nu) : N/A")

        return "\n".join(lines)

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
                "target_names_": self.target_names_,
                "n_targets_": self.n_targets_,
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

        model = cls(config=state["hyperparams"])
        for key, value in state["fitted"].items():
            setattr(model, key, value)

        model._log(f"[INFO] Model loaded from {filepath}")
        return model