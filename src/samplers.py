import math
import numpy as np
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_correlated": GaussianSampler1
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, order=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.order = order

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        
        # sort the samples from the farthest to closest to the last sample.
        # if self.order !=None:
        #     # print('sort samples')
        #     norm = ((xs_b - xs_b[-1, :, :])**2).norm(dim=(1,2))
        #     norm = norm[:(len(norm)-1)]
        #     a = torch.argsort(norm,descending=True)
            # xs_b[:len(a),:,:] = xs_b[a,:,:]
            # xs_b = xs_b**2

        return xs_b

class GaussianSampler1(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, order=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.order = order

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        np.random.seed(42)
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        
        covariance = np.random.exponential(size=(self.n_dims,self.n_dims))*0
        rho = 3  # Adjust this value to control correlation
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                if i > j:
                    covariance[i,j] = rho * np.random.exponential()

        
        covariance = (covariance + covariance.transpose())*0.05 + np.eye(self.n_dims)

        covariance = np.linalg.cholesky(covariance).transpose()
        xs_b = xs_b @ covariance
        for i in range(n_points-1):
            xs_b[:,i+1] = xs_b[:,i+1]/np.sqrt(4)-xs_b[:,i]/np.sqrt(4)*np.sqrt(3)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        
        # sort the samples from the farthest to closest to the last sample.
        # if self.order !=None:
        #     # print('sort samples')
        #     norm = ((xs_b - xs_b[-1, :, :])**2).norm(dim=(1,2))
        #     norm = norm[:(len(norm)-1)]
        #     a = torch.argsort(norm,descending=True)
            # xs_b[:len(a),:,:] = xs_b[a,:,:]
            # xs_b = xs_b**2

        # Confirm the xs are indeed correlated
        xs_b_np = xs_b.numpy()
        xs_b_reshaped = xs_b_np.reshape(b_size * n_points, self.n_dims)
        correlation_matrix = np.corrcoef(xs_b_reshaped, rowvar=False)
        #print("sampled correlation_matrix:")
        #print(correlation_matrix)




        return xs_b.float()
