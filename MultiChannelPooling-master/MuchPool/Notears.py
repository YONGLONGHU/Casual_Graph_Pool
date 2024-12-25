import numpy as np
import scipy.optimize as sopt


def thresholded_causal_matrix(W_est, w_threshold=0.0, percentile=80):
    """
    This function processes the W_est matrix and assigns:
    - The top 80% absolute values of non-zero elements to 1
    - The bottom 20% absolute values to 0
    - All original zero elements remain 0.

    :param W_est: The input matrix (numpy array)
    :param w_threshold: Minimum threshold to consider an element as non-zero
    :param percentile: Percentile threshold (default 80%)
    :return: Thresholded causal matrix
    """
    # 1. Mask non-zero elements
    non_zero_mask = np.abs(W_est) > w_threshold
    W_non_zero = W_est[non_zero_mask]

    if len(W_non_zero) == 0:
        return np.zeros_like(W_est)  # If no non-zero elements, return a zero matrix

    # 2. Sort the absolute values of the non-zero elements
    sorted_values = np.sort(np.abs(W_non_zero))

    # 3. Calculate the threshold for the top 80%
    threshold_value = np.percentile(sorted_values, percentile)

    # 4. Assign values based on the threshold
    causal_matrix = np.zeros_like(W_est)
    causal_matrix[non_zero_mask] = (np.abs(W_est[non_zero_mask]) >= threshold_value).astype(int)

    return causal_matrix
class NotearsWithMask:
    def __init__(self, lambda1=0.1, loss_type='l2', max_iter=100, w_threshold=0):
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.causal_matrix = None

    def learn(self, data, mask):
        X = np.asarray(data)  # Shape (n_samples, n_features)
        M = np.asarray(mask)  # Shape (n_samples, n_samples)

        # Ensure M shape is correct for (n_samples, n_samples)
        assert M.shape == (X.shape[0], X.shape[0])

        # Optimize W using notears_with_mask function
        W_est = self.notears_with_mask(X, M, lambda1=self.lambda1,
                                       loss_type=self.loss_type,
                                       max_iter=self.max_iter)

        # Threshold to create binary causal matrix
        causal_matrix = thresholded_causal_matrix(W_est, w_threshold=0.0, percentile=20)
        self.causal_matrix = causal_matrix
        return causal_matrix

    def notears_with_mask(self, X, M, lambda1, loss_type, max_iter):
        def _loss(W):
            """Evaluate value and gradient of the loss function."""
            W = W * M  # Apply mask to constrain optimization to prior structure.
            # Modify matrix multiplication, making sure the dimensions are correct
            M_pred = W @ X  # X: (n_samples, n_features), W: (n_samples, n_samples)

            # Compute loss and gradient based on selected loss type
            if loss_type == 'l2':
                R = X - M_pred  # Prediction error
                loss = 0.5 / X.shape[0] * np.sum(R ** 2)  # MSE loss
                G_loss = -1.0 / X.shape[0] *  R @ X.T # Gradient of MSE
            else:
                raise ValueError('Unsupported loss type.')

            # Add L1 regularization
            loss += lambda1 * np.sum(np.abs(W))
            G_loss += lambda1 * np.sign(W)  # Gradient of L1 regularization
            return loss, G_loss

        n_samples, n_features = X.shape
        W_est = M  # Shape should be (n_samples, n_samples)

        # Optimization loop
        for i in range(max_iter):
            def _func(w):
                W = w.reshape(n_samples, n_samples)  # Ensure reshaping to (n_samples, n_samples)
                loss, grad = _loss(W)
                return loss, grad.ravel()

            sol = sopt.minimize(_func, W_est.ravel(), method='L-BFGS-B', jac=True)
            W_est = sol.x.reshape(n_samples, n_samples)  # Ensure the reshaped W_est

        return W_est * M  # Apply mask to ensure result respects mask
