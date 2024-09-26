import nanopq
import numpy as np

# W = np.random.randn(2048, 1024).astype(np.float32)
# X = np.random.randn(2048, 1024).astype(np.float32)
W = np.random.random((2048, 1024)).astype(np.float32)
X = np.random.random((2048, 1024)).astype(np.float32)
# Xt = np.random.random((2000, 12)).astype(np.float32)
# query = np.random.random((12, )).astype(np.float32)

pq = nanopq.PQ(M=256, Ks=64, verbose=True)
# pq.fit(vecs=Xt, iter=20, seed=123)
pq.fit(vecs=W, iter=20, seed=123)

W_code = pq.encode(vecs=W)
W_reconstructed = pq.decode(codes=W_code)

relative_error = np.mean(np.abs((W - W_reconstructed) / (W + 1e-9)))
print(f"Relative quantization error: {relative_error:.6f}")

original_result = np.dot(X, W.T)
reconstructed_result = np.dot(X, W_reconstructed.T)
matrix_multiply_error = np.mean(np.abs((original_result - reconstructed_result) / (original_result + 1e-9)))
print(f"Relative error of matrix multiplication: {matrix_multiply_error:.6f}")

print(W[:3])
print(W_reconstructed[:3])

print(original_result[:3])
print(reconstructed_result[:3])
