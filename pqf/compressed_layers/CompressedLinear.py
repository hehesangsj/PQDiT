# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pqf.compression.coding import decode, get_num_centroids
from pqf.compression.kmeans import kmeans
from pqf.compression.kmeans_sr import src
from pqf.compressed_layers.AbstractCompressedLayer import AbstractCompressedLayer


class CompressedLinear(AbstractCompressedLayer):
    """Compressed representation of a linear layer"""

    def __init__(self, codes_matrix: torch.Tensor, codebook: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super(CompressedLinear, self).__init__()

        self.initialize_codes(codes_matrix, codebook)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        self.codebook = nn.Parameter(codebook)

    def _get_uncompressed_weight(self):
        return decode(self.codes_matrix, self.codebook).float()

    def forward(self, x):
        return F.linear(input=x, weight=self._get_uncompressed_weight(), bias=self.bias)

    @staticmethod
    def from_uncompressed(
        uncompressed_layer: torch.nn.Linear,
        k: int,
        k_means_n_iters: int,
        kmeans_fn: Callable,
        subvector_size: int,
        name: str = "",
    ) -> "CompressedLinear":
        """Given an uncompressed layer, initialize the compressed equivalent according to the specified parameters

        Parameters:
            uncompressed_layer: Linear layer to compress
            k: Size of the codebook
            k_means_n_iters: Number of iterations of k means
            subvector_size: Subvector size for the layer
            name : Name of the layer to print alongside mean-squared error
        Returns:
            compressed_layer: Initialized compressed layer
        """

        assert kmeans_fn in [kmeans, src]

        weight = uncompressed_layer.weight.detach()

        c_out, c_in = weight.size()

        num_blocks_per_row = c_in // subvector_size

        training_set = weight.reshape(-1, subvector_size)

        num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)

        codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=k_means_n_iters, slow_cb_update=True)
        codes_matrix = codes.reshape(-1, num_blocks_per_row)

        # Log quantization error
        decoded_weights = decode(codes_matrix, codebook)
        error = (decoded_weights - weight).pow(2).sum() / (num_blocks_per_row * weight.size(0))
        AbstractCompressedLayer.log_quantization_error(name, k_means_n_iters, error, codebook, codes_matrix)

        return CompressedLinear(codes_matrix, codebook, uncompressed_layer.bias)

    def reset_code(self, uncompressed_weight: torch.Tensor):
        weight = uncompressed_weight.detach()
        training_set = weight.reshape(-1, self.codebook.shape[1])
        d = torch.sum(training_set**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', training_set, self.codebook.T)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=1).view(weight.shape[0], -1).byte(), requires_grad=False).to(uncompressed_weight.device)
    
    def get_loss(self, uncompressed_weight: torch.Tensor):
        weight = uncompressed_weight.detach()
        training_set = weight.reshape(-1, self.codebook.shape[1])
        d = torch.sum(training_set**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', training_set, self.codebook.T)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=1).view(weight.shape[0], -1).byte(), requires_grad=False)

        weight_quantized = decode(self.codes_matrix, self.codebook)
        codebook_loss = torch.mean((weight_quantized - weight) **2)

        return codebook_loss

    def reset_code_act(self, uncompressed_weight: torch.Tensor, act: torch.Tensor):
        weight = uncompressed_weight.detach()  # Shape: [C, D]
        training_set = weight.reshape(-1, self.codebook.shape[1])  
        x = act  # Shape: [B, N, C]
        wx = torch.einsum('bnc,dc->bnd', x, training_set)  # xW, shape: [B, N, D]
        wx_codebook = torch.einsum('bnc,dc->bnd', x, self.codebook)  # xW' (codebook), shape: [B, N, D]
        d = torch.sum(wx**2, dim=2, keepdim=True) + \
            torch.sum(wx_codebook**2, dim=2) - 2 * torch.einsum('bnd,bnd->bnd', wx, wx_codebook)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=2).byte(), requires_grad=False).to(uncompressed_weight.device)
    
    def get_loss_act(self, uncompressed_weight: torch.Tensor, act: torch.Tensor):
        weight = uncompressed_weight.detach()
        training_set = weight.reshape(-1, self.codebook.shape[1])
        wx = torch.einsum('bnc,dc->bnd', act, training_set)  # xW, shape: [B, N, D]
        wx_codebook = torch.einsum('bnc,dc->bnd', act, self.codebook)  # xW' (codebook), shape: [B, N, D]
        # Compute L2 distance ||xW - xW'||^2
        d = torch.sum(wx**2, dim=2, keepdim=True) + \
            torch.sum(wx_codebook**2, dim=2) - 2 * torch.einsum('bnd,bnd->bnd', wx, wx_codebook)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=2).byte(), requires_grad=False).to(uncompressed_weight.device)
        codebook_loss = torch.mean((wx - wx_codebook) ** 2)
        return codebook_loss
