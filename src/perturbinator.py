import torch
import torch.nn as nn


class SimpleSMILESEncoder(nn.Module):
    """
    A toy SMILES encoder that one-hot encodes each character
    and processes it via a small MLP or CNN. Here we use a trivial MLP
    for demonstration.
    """

    def __init__(self, vocab, embed_dim=64, hidden_dim=128):
        """
        Args:
            vocab (str or list): A string or list of allowed SMILES characters.
            embed_dim (int): Size of the per-character embedding.
            hidden_dim (int): Size of the hidden layer in the MLP that encodes the SMILES.
        """
        super(SimpleSMILESEncoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embed_dim = embed_dim

        # A simple embedding layer for each character:
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Then a small MLP to compress all characters into a single vector
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # A final pooling transformation to get a single vector
        # (In practice, you might do average/attention pooling across characters.)
        # For simplicity, we apply MLP to each character embedding and average.
        # More advanced: CNN or RNN or Transformer.

    def forward(self, smiles_batch):
        tokens, _ = self.smiles_to_indices(smiles_batch)

        # Make sure tokens are on the same device as the embedding layer
        tokens = tokens.to(self.embedding.weight.device)

        embedded = self.embedding(tokens)
        B, L, E = embedded.shape
        embedded = embedded.view(B * L, E)
        embedded = self.mlp(embedded)
        embedded = embedded.view(B, L, -1)
        embedded = embedded.mean(dim=1)
        return embedded

    def smiles_to_indices(self, smiles_batch):
        """
        Convert list of SMILES strings to a batch of indices for each character.
        We'll simply pad them to max_len in this toy example.
        """
        # Find max length
        max_len = max(len(s) for s in smiles_batch)
        token_ids = []
        lengths = []

        char_to_idx = {c: i for i, c in enumerate(self.vocab)}

        for s in smiles_batch:
            lengths.append(len(s))
            row = []
            for ch in s:
                if ch in char_to_idx:
                    row.append(char_to_idx[ch])
                else:
                    row.append(char_to_idx["?"])
            # pad
            while len(row) < max_len:
                row.append(char_to_idx[" "])
            token_ids.append(row)

        # Convert to torch tensor
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return tokens, lengths


class Perturbinator(nn.Module):
    """
    A model that encodes (unperturbed gene expression) + (drug SMILES)
    and predicts (perturbed gene expression).
    """

    def __init__(
        self, gene_dim, gene_hidden_dim=512, drug_hidden_dim=128, smiles_vocab=None
    ):
        """
        Args:
            gene_dim (int): Dimensionality of the gene expression input.
            gene_hidden_dim (int): hidden dimension for the gene MLP.
            drug_hidden_dim (int): hidden dimension for the SMILES encoder output.
            smiles_vocab (str or list): vocabulary for the SMILES.
        """
        super(Perturbinator, self).__init__()
        if smiles_vocab is None:
            # A simple default vocabulary of typical SMILES chars + space + ?
            # In practice, define a more robust set of tokens
            smiles_vocab = "ACGT()[]=+#@0123456789abcdefghijklmnopqrstuvwxyz" + "? "

        self.gene_dim = gene_dim
        self.gene_hidden_dim = gene_hidden_dim
        self.drug_hidden_dim = drug_hidden_dim

        # 2.1) Gene MLP
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, gene_hidden_dim),
            nn.ReLU(),
            nn.Linear(gene_hidden_dim, gene_hidden_dim),
            nn.ReLU(),
        )

        # 2.2) SMILES encoder
        self.smiles_encoder = SimpleSMILESEncoder(
            vocab=smiles_vocab, embed_dim=64, hidden_dim=drug_hidden_dim
        )

        # 2.3) Fusion -> final MLP to predict the same dimension as gene_dim
        self.fusion = nn.Sequential(
            nn.Linear(gene_hidden_dim + drug_hidden_dim, gene_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                gene_hidden_dim, gene_dim
            ),  # we predict perturbed expression (size = gene_dim)
        )

    def forward(self, gene_expr, smiles_batch):
        """
        gene_expr: Tensor of shape (B, gene_dim)
        smiles_batch: List of length B containing SMILES strings
        """
        # Encode gene
        gene_emb = self.gene_encoder(gene_expr)  # (B, gene_hidden_dim)
        # Encode SMILES
        drug_emb = self.smiles_encoder(smiles_batch)  # (B, drug_hidden_dim)

        fused = torch.cat(
            [gene_emb, drug_emb], dim=1
        )  # (B, gene_hidden_dim + drug_hidden_dim)
        out = self.fusion(fused)
        return out
