import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Codebook(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(Codebook, self).__init__()
        self.codes = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, features):
        # Calculate distances between features and codebook entries
        dists = torch.cdist(features, self.codes)
        # Find the nearest code
        indices = torch.argmin(dists, dim=1)
        # Retrieve the corresponding codes
        return self.codes[indices]

class VQCodebook(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, initial_vectors=None):
        super(VQCodebook, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.encoder = Encoder(3,128,2,32)
        self.pre_quantization_conv = nn.Conv2d(
            128, embedding_dim, kernel_size=1, stride=1)
        self.decoder = Decoder(embedding_dim,128,2,32)

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1), # 32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64,16,16
        )
        
        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1) # 32, 32, 32
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64,16,16
        self.decoder_conv1 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1, 1]로 정규화된 이미지를 얻기 위해 사용 (선택 사항)
        )"""

    def forward(self, patch, idx=None):
        z = self.encoder(patch) # 32, 128, 16, 16
        z = self.pre_quantization_conv(z) # 32, 64, 16, 16
        z = z.permute(0, 2, 3, 1).contiguous() # 32, 16, 16, 64
        
        if idx is not None:
            print(f"Initializing codebook at index {idx}")
            for i in range(32): # batch_size: 32
                self.initialize_embedding_with_vector(self.embedding, z[i], idx * 32 + i)
        

        # Flatten input using reshape instead of view
        z_flattened = z.reshape(-1, self.embedding_dim) # 8192, 64

        # Calculate distances between z and embedding
        dists = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # Get the closest codebook entry
        min_encoding_indices = torch.argmin(dists, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantize the input using the closest codebook entry
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Calculate VQ Losses
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous() # 32, 64, 16, 16
        reconstructed_patch = self.decoder(z_q)
        return reconstructed_patch, codebook_loss, commitment_loss, min_encoding_indices #encoding_indices
    """
    def encode_patch(self, patch):
        output = F.relu(self.encoder_conv1(patch))
        output = self.encoder_conv2(output)
        return output  
    def decode_vector(self, vector):
        #vector = vector.unsqueeze(2).unsqueeze(3)
        #vector = nn.functional.interpolate(vector, scale_factor=(64, 64), mode='bilinear', align_corners=False)  # [batch_size, embedding_dim, 64, 64]
        output = F.relu(self.decoder_conv1(vector))
        output = self.decoder_conv2(output)       
        
        return output  # [batch_size, patch_size, patch_size]
        """

    def initialize_embedding_with_vectors(self, embedding_layer, initial_vectors):
        # Check if the shape matches
        if embedding_layer.weight.shape != initial_vectors.shape:
            raise ValueError(f"Shape of initial_vectorss {initial_vectors.shape} does not match "
                             f"embedding layer weight shape {embedding_layer.weight.shape}")

        # Initialize the embedding weights with the given vectors
        with torch.no_grad():
            embedding_layer.weight.copy_(initial_vectors)

    def initialize_embedding_with_vector(self, embedding_layer, vector, index):
        # Check if the input vector has the correct dimension
        if vector.shape[0] != embedding_layer.embedding_dim:
            raise ValueError(f"The input vector should have shape [{embedding_layer.embedding_dim}], "
                             f"but got shape {vector.shape}")

        # Initialize the specific embedding at the given index
        with torch.no_grad():
            embedding_layer.weight[index].copy_(vector)