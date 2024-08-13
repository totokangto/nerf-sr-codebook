import torch
import torch.nn as nn

# self.codebook = Codebook(opt.code_size, opt.num_codes).to(self.device)
# codebook_features = self.netEnhancer(self.data_sr_patch, data_ref_patches)
# generated_patches = self.codebook(codebook_features)

class Codebook(nn.Module):
    def __init__(self, code_size, num_codes):
        super(Codebook, self).__init__()
        self.codes = nn.Parameter(torch.randn(num_codes, code_size))

    def forward(self, features):
        # Calculate distances between features and codebook entries
        dists = torch.cdist(features, self.codes)
        # Find the nearest code
        indices = torch.argmin(dists, dim=1)
        # Retrieve the corresponding codes
        return self.codes[indices]

class VQCodebook(nn.Module):
    def __init__(self, code_size, num_codes):
        super(VQCodebook, self).__init__()
        self.num_codes = num_codes
        self.code_size = code_size
        self.embedding = nn.Embedding(num_codes, code_size)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, z):
        # Flatten input
        z_flattened = z.view(-1, self.code_size)

        # Calculate distances between z and embedding
        dists = torch.cdist(z_flattened, self.embedding.weight)
        
        # Get the closest codebook entry
        encoding_indices = torch.argmin(dists, dim=1).unsqueeze(1)

        # Quantize the input using the closest codebook entry
        z_q = self.embedding(encoding_indices).view_as(z)

        # Calculate VQ Losses
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        
        return z_q, codebook_loss, commitment_loss, encoding_indices
