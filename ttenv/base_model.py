from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.linalg import solve_triangular, cholesky
from scipy.stats import norm, multivariate_normal
from scipy.stats import poisson


def entropy_multivariate_gaussian(cov_matrix: np.ndarray) -> float:
    """
    Calculate the entropy of a multivariate Gaussian distribution.

    :param cov_matrix: Covariance matrix of the distribution (2D numpy array).
    :return: Entropy of the distribution.
    """
    k = 1  # Number of dimensions
    if cov_matrix.ndim > 1:
        k = cov_matrix.shape[0]
        det_cov_matrix = np.linalg.det(cov_matrix)  # Determinant of the covariance matrix
    else:
        det_cov_matrix = cov_matrix
    # Ensure the determinant is positive
    if det_cov_matrix <= 0:
        raise ValueError("Covariance matrix must have a positive determinant")

    entropy = 0.5 * np.log((2 * np.pi * np.e) ** k * det_cov_matrix)

    return np.squeeze(entropy)


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Bhattacharyya distance between two normal distributions.

    Supports both univariate and multivariate normal distributions.

    Parameters:
    mu1 (array-like): Mean of the first normal distribution.
    sigma1 (array-like): Covariance matrix (or standard deviation) of the first normal distribution.
    mu2 (array-like): Mean of the second normal distribution.
    sigma2 (array-like): Covariance matrix (or standard deviation) of the second normal distribution.

    Returns:
    float: The Bhattacharyya distance between the two normal distributions.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    sigma1 = np.asarray(sigma1)
    sigma2 = np.asarray(sigma2)

    if mu1.ndim == 0 or len(mu1) == 1:
        # Univariate case
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Standard deviations must be positive for univariate normal distributions")

        term1 = 0.25 * ((mu1 - mu2) ** 2) / (sigma1 ** 2 + sigma2 ** 2)
        term2 = 0.5 * np.log((sigma1 ** 2 + sigma2 ** 2) / (2 * sigma1 * sigma2))

        return term1 + term2
    else:
        # Multivariate case
        if mu1.shape != mu2.shape:
            raise ValueError("Means must have the same dimensions")
        if sigma1.shape != sigma2.shape or sigma1.shape[0] != sigma1.shape[1]:
            raise ValueError("Covariance matrices must be square and have the same dimensions")

        # Calculate the average of the covariance matrices
        sigma_avg = (sigma1 + sigma2) / 2.0

        # Calculate the Mahalanobis distance part
        diff = mu2 - mu1
        mahalanobis_term = 0.125 * diff.T.dot(np.linalg.inv(sigma_avg)).dot(diff)

        # Calculate the determinant part
        det_term = 0.5 * np.log(np.linalg.det(sigma_avg) / np.sqrt(np.linalg.det(sigma1) * np.linalg.det(sigma2)))

        bd = mahalanobis_term + det_term
        return np.squeeze(bd)


def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two normal distributions.

    Supports both univariate and multivariate normal distributions.

    Parameters:
    mu1 (array-like): Mean of the first normal distribution.
    sigma1 (array-like): Covariance matrix (or standard deviation) of the first normal distribution.
    mu2 (array-like): Mean of the second normal distribution.
    sigma2 (array-like): Covariance matrix (or standard deviation) of the second normal distribution.

    Returns:
    float: The KL divergence between the two normal distributions.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    sigma1 = np.asarray(sigma1)
    sigma2 = np.asarray(sigma2)

    if mu1.ndim == 0 or len(mu1) == 1:
        # Univariate case
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Standard deviations must be positive for univariate normal distributions")

        term1 = np.log(sigma2 / sigma1)
        term2 = (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        term3 = -0.5

        return term1 + term2 + term3
    else:
        # Multivariate case
        k = mu1.shape[0]

        if sigma1.shape != (k, k) or sigma2.shape != (k, k):
            raise ValueError("Covariance matrices must be square and match the dimensionality of the means")

        term1 = np.trace(np.linalg.inv(sigma2).dot(sigma1))
        term2 = (mu2 - mu1).T.dot(np.linalg.inv(sigma2)).dot(mu2 - mu1)
        term3 = -k
        term4 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
        kl = 0.5 * (term1 + term2 + term3 + term4)
        return np.squeeze(kl)


class GaussianDistribution:
    def __init__(self, mean, var) -> None:
        self.mean = mean
        self.var = var

    def sample(self, n):
        if np.ndim(self.mean) > 0:
            return np.random.multivariate_normal(self.mean, self.var, n)
        else:
            return np.random.normal(self.mean, np.sqrt(self.var), n)

    def pdf(self, x):
        if np.ndim(self.mean) > 0:
            return multivariate_normal.pdf(x, self.mean, self.var)
        else:
            return norm.pdf(x, self.mean, np.sqrt(self.var))

    def log_pdf(self, x):
        if np.ndim(self.mean) > 0:
            return multivariate_normal.logpdf(x, self.mean, self.var)
        else:
            return norm.logpdf(x, self.mean, np.sqrt(self.var))


class GMMDist:
    def __init__(self, weights, means, covs) -> None:
        self.means = np.array(means)
        self.covs = np.array(covs)
        self.weights = np.array(weights)
        self.n_components = len(weights)

    def compute_mms(self):
        num_of_comp = len(self.means)
        if np.ndim(self.means) > 1:
            mean_val = np.matmul(self.weights, self.means)
        else:
            mean_val = np.dot(self.weights, self.means)
        size_of_mean = self.means.shape
        if np.ndim(self.means) <= 1:
            cov_val = 0
        else:
            cov_val = np.zeros((size_of_mean[1], size_of_mean[1]))
        for i in range(num_of_comp):
            cov_val += self.weights[i] * (self.covs[i] + np.einsum('i,j->ij', self.means[i], self.means[i]))

        cov_val -= np.einsum('i,j->ij', mean_val, mean_val)
        cov_val = cov_val.squeeze()
        return mean_val, cov_val

    def sg_entropy_ub(self):
        m, v = self.compute_mms()
        if np.ndim(v) == 0:
            ub = 0.5 * np.log(2 * np.pi * np.e * v)
        else:
            k = v.shape[0]  # Number of dimensions
            determinant = np.linalg.det(v)
            ub = 0.5 * (k + k * np.log(2 * np.pi) + np.log(determinant))
        return ub

    def sample(self, n_samples):
        samples = []
        component_samples = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        for i in range(self.n_components):
            n_component_samples = np.sum(component_samples == i)
            if n_component_samples > 0:
                if len(self.means[0]) > 1:
                    samples.append(
                        np.random.multivariate_normal(self.means[i], self.covs[i], n_component_samples))
                else:
                    samples = np.concatenate(
                        (samples, np.random.normal(self.means[i], self.covs[i].squeeze(), n_component_samples)))
        if len(self.means[0]) > 1:
            return np.vstack(samples)
        else:
            return samples

    def entropy_mc_estimate(self, n_samples):
        samples = self.sample(n_samples)
        pdfs = self.pdf(samples)
        ent = np.mean(-np.log(pdfs))
        return ent

    #
    def pdf(self, x):
        if np.ndim(self.covs) <= 1:
            prob = np.dot(self.weights, np.squeeze(
                [norm.pdf(x, self.means[i], np.sqrt(self.covs[i]))
                 for i in range(len(self.weights))]))
        else:
            if np.ndim(x) == 1:
                prob = np.dot(self.weights, np.array(
                    [multivariate_normal.pdf(x, self.means[i], self.covs[i])
                     for i in range(len(self.weights))]))
            else:
                prob = np.matmul(self.weights,
                                 [multivariate_normal.pdf(x, self.means[i], self.covs[i])
                                  for i in range(len(self.weights))])
        return prob

    def pwd_entropy_estimate(self):
        h_cond = 0
        for i in range(self.n_components):
            h_cond += self.weights[i] * entropy_multivariate_gaussian(self.covs[i])
        low_term = 0
        up_term = 0
        for i in range(self.n_components):
            low_add = 0
            up_add = 0
            for j in range(self.n_components):
                if np.ndim(self.means[i]) == 0 or len(self.means[i]) == 1:
                    bc_dist = bhattacharyya_distance(self.means[i], np.sqrt(self.covs[i]), self.means[j],
                                                     np.sqrt(self.covs[j]))
                    kl = kl_divergence_normal(self.means[i], np.sqrt(self.covs[i]), self.means[j],
                                              np.sqrt(self.covs[j]))
                else:
                    bc_dist = bhattacharyya_distance(self.means[i], self.covs[i], self.means[j],
                                                     self.covs[j])
                    kl = kl_divergence_normal(self.means[i], self.covs[i], self.means[j],
                                              self.covs[j])
                low_add += self.weights[j] * np.exp(-bc_dist)
                up_add += self.weights[j] * np.exp(-kl)
            low_term += self.weights[i] * np.log(low_add)
            up_term += self.weights[i] * np.log(up_add)
        return np.squeeze(h_cond - low_term), np.squeeze(h_cond - up_term)
    #
    # def log_pdf(self, x):
    #     return norm.logpdf(x, self.mean, np.sqrt(self.var))


class LinearGaussianDistribution:
    # y ~ N( y | Ax+b,C)
    def __init__(self, coefficient, noise_mean, noise_var) -> None:
        self.coefficient = coefficient
        self.noise_mean = noise_mean
        self.noise_var = noise_var

    # sample y given realizations of x
    def sample(self, x, n):
        if np.ndim(self.noise_mean) > 0:
            if np.ndim(x) == 1:
                return np.random.multivariate_normal(np.matmul(self.coefficient, x) + self.noise_mean, self.noise_var,
                                                     n).squeeze()
            else:
                return np.array([
                    np.random.multivariate_normal(np.matmul(self.coefficient, x[i]) + self.noise_mean, self.noise_var,
                                                  1) for i in range(n)]).squeeze()
        else:
            return np.random.normal(self.coefficient * x + self.noise_mean, np.sqrt(self.noise_var), n)

    def pdf(self, y, x):
        if np.ndim(self.noise_mean) > 0:
            return multivariate_normal.pdf(y, np.matmul(self.coefficient, x) + self.noise_mean, self.noise_var)
        else:
            return norm.pdf(y, self.coefficient * x + self.noise_mean, np.sqrt(self.noise_var))

    def log_pdf(self, y, x):
        if np.ndim(self.noise_mean) > 0:
            return multivariate_normal.logpdf(y, np.matmul(self.coefficient, x) + self.noise_mean, self.noise_var)
        else:
            return norm.logpdf(y, self.coefficient * x + self.noise_mean, np.sqrt(self.noise_var))

    # compute mean and variance/cov of p(Y)
    def compute_marginal_moments(self, x_mean, x_var):

        if np.ndim(x_var) <= 1:
            mu_marg = self.coefficient * np.array(x_mean) + self.noise_mean
            var_marg = np.array(self.noise_var) + self.coefficient * np.array(x_var) * np.array(self.coefficient).T
            if np.ndim(var_marg) < 1 and np.ndim(mu_marg) == 1:
                var_marg = np.array([var_marg] * len(mu_marg))
        else:
            mu_marg = np.matmul(self.coefficient, np.array(x_mean)) + self.noise_mean
            np_x_var = np.array(x_var)
            # if np_x_var.shape[0] == np_x_var.shape[1]:
            #     var_marg = np.array(self.noise_var) + np.matmul(np.array(
            #         self.coefficient).T, np.matmul(self.coefficient, np_x_var))
            #     # var_marg = np.array(self.noise_var) + self.coefficient * x_var * np.array(
            #     #     self.coefficient).T
            # else:
            #     var_marg = np.array(self.noise_var) + self.coefficient * np.array(x_var) * np.array(self.coefficient).T
            var_marg = np.array(self.noise_var) + np.matmul(np.array(
                self.coefficient).T, np.matmul(self.coefficient, np_x_var))
        return mu_marg, var_marg

    def compute_marginal_mean(self, x_mean):
        if np.ndim(self.noise_mean) > 0:
            mu_marg = np.matmul(self.coefficient, x_mean.T).T + self.noise_mean
            return mu_marg
        else:
            mu_marg = self.coefficient * np.array(x_mean) + self.noise_mean
            return mu_marg

    def compute_marginal_dists(self, x_mean, x_var):
        mu_marg, var_marg = self.compute_marginal_moments(x_mean, x_var)
        n1 = len(mu_marg) if np.ndim(mu_marg) > 0 else 1
        n2 = len(var_marg) if np.ndim(var_marg) > 0 else 1
        if n1 == n2 != 1:
            return [GaussianDistribution(mu_marg[i], var_marg[i]) for i in range(0, n1)]
        elif n1 == n2 == 1:
            return [GaussianDistribution(mu_marg, var_marg)]
        elif n1 == 1 and n2 != 1:
            return [GaussianDistribution(mu_marg, var_marg[i]) for i in range(0, n2)]
        elif n2 == 1 and n1 != 1:
            return [GaussianDistribution(mu_marg[i], var_marg) for i in range(0, n1)]
        else:
            raise ValueError("Length error")

    # compute p(Y=y|X)
    def compute_marginal_likelihood(self, y, x_mean, x_var):
        mu_marg, var_marg = self.compute_marginal_moments(x_mean, x_var)
        if np.ndim(x_mean) > 0:
            if np.ndim(y) > 1:
                pdf_vals = [multivariate_normal.pdf(y_item, mu_marg, np.sqrt(var_marg)) for y_item in y]
                return np.squeeze(pdf_vals)
            else:
                return multivariate_normal.pdf(y, mu_marg, np.sqrt(var_marg))
        else:
            if np.ndim(y) > 0:
                pdf_vals = [norm.pdf(y_item, mu_marg, np.sqrt(var_marg)) for y_item in y]
                return np.squeeze(pdf_vals)
            else:
                return norm.pdf(y, mu_marg, np.sqrt(var_marg))

    # compute mean and variance/cov of p(X|Y)
    def compute_post_moments(self, y, x_mean, x_var):
        x_mean_np = np.array(x_mean)
        x_var_np = np.array(x_var)
        x_var_inv = 1 / x_var_np if np.ndim(x_var) <= 1 or x_var_np.shape[0] != x_var_np.shape[1] else np.linalg.inv(
            x_var_np)
        y_np = np.array(y)
        coef_np = np.array(self.coefficient)
        r_inv = 1 / np.array(self.noise_var) if np.ndim(self.noise_var) <= 1 else np.linalg.inv(self.noise_var)
        post_pre = x_var_inv + coef_np.T * r_inv * coef_np
        post_var = 1 / post_pre if np.ndim(post_pre) <= 1 or post_pre.shape[0] != post_pre.shape[1] else np.linalg.inv(
            post_pre)
        post_mean = post_var * (coef_np.T * r_inv * (y_np - self.noise_mean) + x_var_inv * x_mean_np)
        return post_mean, post_var


class LinearGaussianMixtureModel:
    def __init__(self, weights, coefficients, noise_means, noise_vars) -> None:
        self.weights = weights
        self.noise_means = noise_means
        self.noise_vars = noise_vars
        self.coefficients = coefficients
        self.num_of_comp = len(weights)
        self.linear_gaussians = []
        for i in range(0, self.num_of_comp):
            self.linear_gaussians.append(LinearGaussianDistribution(coefficients[i], noise_means[i], noise_vars[i]))

    def display(self):
        # print("weights:")
        print("[" + ";".join(map(str, self.weights)) + "]")
        print(",")
        # print("coefficients:")
        # print(self.coefficients)
        # print("means:")
        # print(self.noise_means)
        # print("vars:")
        print("[" + ";".join(map(str, self.noise_vars)) + "]")

    def sample(self, x, n):
        assignments = np.random.choice(self.num_of_comp, n, p=self.weights)
        if n > 1:
            samples = []
            for i in range(n):
                mode = assignments[i]
                samples.append(self.linear_gaussians[mode].sample(x[i], 1))
        else:
            samples = [self.linear_gaussians[assignments[0]].sample(x, n)]
        return np.squeeze(samples)

    def pdf(self, y, x):
        probs = np.zeros(np.shape(y))
        for i in range(0, self.num_of_comp):
            probs += self.linear_gaussians[i].pdf(y, x) * self.weights[i]
        return probs

    def log_pdf(self, y, x):
        return np.log(self.pdf(y, x))

    def compute_marginal_moments(self, prior_mean, prior_var):
        results = [[], []]
        for i in range(0, self.num_of_comp):
            m, v = self.linear_gaussians[i].compute_marginal_moments(prior_mean, prior_var)
            if np.ndim(prior_mean) > 1:
                if i == 0:
                    results[0] = m
                    results[1] = v
                else:
                    results[0] = np.concatenate((results[0], m), axis=0)
                    results[1] = np.concatenate((results[1], v), axis=0)
            else:
                results[0].append(m)
                results[1].append(v)

        return np.array(results[0]).squeeze(), np.array(results[1]).squeeze()

    def compute_marginal_means(self, prior_mean):
        results = [self.linear_gaussians[i].compute_marginal_mean(prior_mean) for i in
                   range(0, self.num_of_comp)]
        return np.array(results)


class ParticleDist:
    def __init__(self, n, weights, x):
        self.n = n
        self.weights = weights
        self.x = x

    def sample(self, n):
        random_samples = np.random.choice(self.x.shape[0], size=n, replace=True, p=self.weights)
        sampled_data = self.x[random_samples]
        return sampled_data


class AIAPos:
    def __init__(self, coord):
        self.coord = coord

    def euclid_dist(self, target):
        dist = np.linalg.norm(np.array(self.coord) - np.array(target.coord))
        return dist

    def cal_angle(self, target):
        if isinstance(target, AIAPos):
            v2 = np.array(target.coord) - np.array(self.coord)
        else:
            v2 = np.array(target) - np.array(self.coord)
        angle_radians = np.arctan2(v2[1], v2[0])
        angle = np.degrees(angle_radians)
        return angle

    @staticmethod
    def euclid_dist_static(origin, target):
        dist = np.linalg.norm(np.array(origin.coord) - np.array(target.coord))
        return dist

    @staticmethod
    def manhattan_dist_static(origin, target):
        dist = abs(origin.coord[0] - target.coord[0]) + abs(origin.coord[1] - target.coord[1])
        return dist


def batch_mvnorm_logpdf(x, mus, Sigma):
    # x:    (d,)
    # mus:  (N, d)
    # Sigma:(d, d)
    d = x.shape[0]
    N = mus.shape[0]

    # 1) Cholesky
    L = cholesky(Sigma, lower=True)  # (d, d)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    logC = -0.5 * (d * np.log(2 * np.pi) + log_det)

    # 2) Residuals
    delta = x - mus  # (N, d)
    # 3) Whiten all: solve L * Z = delta^T
    Z = solve_triangular(L, delta.T, lower=True)  # (d, N)

    # 4) Squared Mahalanobis
    sqnorm = np.sum(Z * Z, axis=0)  # (N,)

    # 5) PDF
    log_pdfs = logC - 0.5 * sqnorm
    return log_pdfs  # (N,)


def batch_mvnorm_logpdf_multi_cov(x, mus, Sigmas):
    """
    x:      (d,) or (1,d)
    mus:    (N,d)
    Sigmas: (N,d,d)
    returns p: (N,)
    """
    x = torch.tensor(x)
    mus = torch.tensor(mus)
    Sigmas = torch.tensor(Sigmas)

    N, d = mus.shape

    # 1) Batched Cholesky: Sigmas -> Ls  where Sigmas[i] = Ls[i] @ Ls[i].T
    Ls = torch.linalg.cholesky(Sigmas)  # (N, d, d)

    # 2) log-determinants
    #    det(Sigma_i) = (prod diag(L_i))^2
    logdets = 2.0 * torch.log(torch.diagonal(Ls, dim1=1, dim2=2)).sum(dim=1)  # (N,)

    # 3) normalization constants:
    #    log C_i = -½ [ d*log(2π) + logdet_i ]
    logCs = -0.5 * (d * torch.log(torch.tensor(2 * torch.pi)) + logdets)  # (N,)

    # 4) residuals & whitening
    #    for each i: solve Ls[i] @ z_i = (x - mu_i).T
    delta = x.unsqueeze(0) - mus  # (N, d)
    # torch.triangular_solve supports batch if inputs are (N,d,1) etc:
    zs = torch.linalg.solve_triangular(
        Ls,  # (N, d, d)
        delta,  # (N, d, 1)
        upper=False,  # Ls is lower-triangular
        left=True,  # solve L @ Z = delta
        unitriangular=False
    )
    zs = zs.squeeze(-1)  # (N, d)

    # 5) squared Mahalanobis norms
    sqnorms = (zs ** 2).sum(dim=1)  # (N,)

    # 6) final PDF values
    log_pdfs = logCs - 0.5 * sqnorms
    return log_pdfs.numpy()


class GasLeakageModel:
    def __init__(self):
        self.tau = 1500
        self.D = 1
        self.V = 0
        self.Q0 = 10
        self.a = 10
        self.t0 = 0.5
        self.rate_threshold = 50
        self.klist = []
        self.pmf_list = []

    def get_emission_rate(self, agent, target):
        lam = np.sqrt(self.D * self.tau / (1 + self.V ** 2 * self.tau / (4 * self.D)))
        # r = self.Q0 / np.log(lam / self.a) * np.exp((agent_coord[0] - target_coord[0]) * self.V / (2 * self.D)) * i0(
        #     AIAPos.euclid_dist_static(AIAPos(agent_coord), AIAPos(target_coord)) / lam)
        dist = np.linalg.norm(np.array(agent[:2]) - np.array(target[:2]))

        if dist == 0:
            r = self.rate_threshold
        else:
            r = self.a * self.Q0 / dist * np.exp(-dist / lam) * np.exp(
                -(target[0] - agent[0]) * self.V / (2 * self.D))
        r = min(self.rate_threshold, r)
        return r

    def get_mus(self, agent_coords, target_coords):
        mu_list = [self.get_mu(a, t) for a, t in zip(agent_coords, target_coords)]
        return np.array(mu_list)

    def get_mu(self, agent_coord, target_coord):
        r = self.get_emission_rate(agent_coord, target_coord)
        mu = r * self.t0
        return mu

    def generate_sample(self, target_coord, agent_coord):
        mu = self.get_mu(agent_coord, target_coord)
        sample = np.random.poisson(mu)
        return np.array(sample)

    def generate_samples(self, agent_coord, target_coord, n):
        mu = self.get_mu(agent_coord, target_coord)
        samples = np.random.poisson(mu, (n, 1))
        return samples

    def generate_samples_pairwise(self, agent_coords, target_coords):
        mus = []
        for a_coord, t_coord in zip(agent_coords, target_coords):
            mu = self.get_mu(a_coord, t_coord)
            mus.append(mu)
        samples = np.random.poisson(mus)
        return samples.reshape([-1, 1])

    def measurement_llhs(self, agent_coords, target_coords, ys):
        probs = [poisson(self.get_mu(a_coord, t_coord)).pmf(ys.squeeze()) for a_coord, t_coord in
                 zip(agent_coords, target_coords)]
        return probs

    def measurement_log_llhs(self, agent, target_coords, y):
        log_probs = [poisson(self.get_mu(agent, t_coord)).logpmf(y) for t_coord in
                     target_coords]
        return log_probs

    def plot_mu(self, target_pos, x_range, y_range, n):
        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], n)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        mus = [self.get_mu(a, target_pos) for a in positions]
        plt.contourf(xx, yy, np.reshape(mus, [n, n]), levels=20, cmap='coolwarm')
        plt.colorbar()
        plt.title('Gas Leakage Mean Concentration')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig("gas leakage measure mu.png")
        plt.show()

    def get_entropy(self, agent_coords, target_coords):
        ents = []
        for a_coord, t_coord in zip(agent_coords, target_coords):
            mu = self.get_mu(a_coord, t_coord)
            ent = poisson(mu).entropy()
            ents.append(ent)
        return ents

    def preheat(self):
        max_lam = self.rate_threshold * self.t0
        prob_threshold = 0.99
        klist = []
        for lam in range(max_lam + 1):
            low, high = 0, int(lam + 10 * (lam ** 0.5))  # Heuristic for upper bound based on lambda
            # Perform binary search
            while low < high:
                mid = (low + high) // 2
                cumulative_prob = poisson.cdf(mid, lam)

                if cumulative_prob < prob_threshold:
                    low = mid + 1  # Move the lower bound up
                else:
                    high = mid  # Narrow down to mid

            klist.append(low)
        self.klist = klist
