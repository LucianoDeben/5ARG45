import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
}

NORM_LAYERS = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm, "none": None}


class FlexibleFCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ):
        super(FlexibleFCNN, self).__init__()
        self.residual = residual

        # Ensure hidden_dims are consistent for residual
        if residual:
            hidden_dims = [hidden_dims[0]] * len(hidden_dims)

        self.activation = ACTIVATIONS.get(activation_fn.lower(), nn.ReLU)()
        self.norm_type = norm_type.lower()

        # Build layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            norm = (
                NORM_LAYERS.get(self.norm_type, nn.Identity)(dims[i + 1])
                if norm_type != "none"
                else nn.Identity()
            )
            self.norms.append(norm)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for layer, norm in zip(self.layers, self.norms):
            identity = out  # Save for residual

            # Main path
            out = layer(out)
            out = norm(out)

            # Optional residual
            if self.residual and (out.shape == identity.shape):
                out = out + identity

            out = self.activation(out)
            out = self.dropout(out)

        return self.output(out)


class SparseKnowledgeNetwork(nn.Module):
    def __init__(
        self,
        gene_tf_matrix: torch.Tensor,
        hidden_dims: list,
        output_dim: int = 1,
        first_activation: str = "tanh",
        downstream_activation: str = "relu",
        dropout_prob: float = 0.2,
        weight_init: str = "xavier",
        use_batchnorm: bool = True,
    ):
        super(SparseKnowledgeNetwork, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm

        # Store the precomputed gene-TF matrix as a trainable parameter
        self.gene_tf_matrix = nn.Parameter(gene_tf_matrix)

        # First activation function
        FIRST_ACTIVATIONS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}
        self.first_activation = FIRST_ACTIVATIONS.get(first_activation.lower())
        if self.first_activation is None:
            raise ValueError("First activation must be 'tanh' or 'sigmoid'.")

        # Downstream activation function
        DOWNSTREAM_ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}
        self.downstream_activation = DOWNSTREAM_ACTIVATIONS.get(
            downstream_activation.lower()
        )
        if self.downstream_activation is None:
            raise ValueError("Unknown downstream activation.")

        # Define the hidden layers after the TF layer
        tf_dim = self.gene_tf_matrix.shape[1]
        hidden_dims = [tf_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method="xavier"):
        for layer in list(self.hidden_layers) + [self.output_layer]:
            if method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight initialization method: {method}")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Gene-to-TF interaction layer
        tf_activations = torch.matmul(x, self.gene_tf_matrix)

        # Apply the first activation function
        tf_activations = self.first_activation(tf_activations)

        # Pass through hidden layers
        hidden_activations = tf_activations
        for i, layer in enumerate(self.hidden_layers):
            hidden_activations = layer(hidden_activations)
            if self.use_batchnorm:
                hidden_activations = self.batch_norms[i](hidden_activations)
            hidden_activations = self.downstream_activation(hidden_activations)
            hidden_activations = self.dropout(hidden_activations)

        # Output layer
        output = self.output_layer(hidden_activations)
        return output


import networkx as nx
import networkx.algorithms.dag as nxadag
import torch
import torch.nn as nn
import torch.nn.functional as F


class genecell_nn(nn.Module):
    """
    An ontology-based neural network for gene expression data.

    This network uses a given ontology (DAG) to guide a hierarchical neural network.
    For each ontology term, a module is built that takes as input the combined signals
    from its child terms (if any) and from the genes that are directly annotated to that term.
    The network is constructed bottom-up so that the root node’s representation is used for
    the final regression prediction.

    Attributes:
        root (str): The root term of the ontology.
        gene_dim (int): The dimension of the gene expression input.
        num_hiddens_genotype (int): The number of hidden neurons used for each term.
        num_hiddens_final (int): The number of neurons in the final fully-connected layer.
        term_direct_gene_map (dict): Mapping from ontology terms to the set of directly annotated gene IDs.
        term_dim_map (dict): Mapping from each ontology term to its hidden dimension.
        term_neighbor_map (dict): Mapping from each ontology term to a list of its children in the ontology.
        term_layer_list (list): A list (ordered bottom-up) of lists of terms (from leaves upward).
    """

    def __init__(
        self,
        term_size_map,
        term_direct_gene_map,
        dG,
        gene_dim,
        root,
        num_hiddens_genotype,
        num_hiddens_final,
    ):
        """
        Initialize the genecell network.

        Args:
            term_size_map (dict): Mapping from ontology terms to total number of genes (direct plus descendants).
            term_direct_gene_map (dict): Mapping from ontology terms to the set of gene IDs directly annotated.
            dG (networkx.DiGraph): The ontology represented as a directed acyclic graph.
            gene_dim (int): The dimensionality of the gene expression input.
            root (str): The root node (term) of the ontology.
            num_hiddens_genotype (int): Number of hidden neurons for each term module.
            num_hiddens_final (int): Number of neurons in the final layer.
        """
        super(genecell_nn, self).__init__()
        self.root = root
        self.gene_dim = gene_dim
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_final = num_hiddens_final
        self.term_direct_gene_map = term_direct_gene_map

        # Set the hidden dimensionality for each term.
        # For simplicity, we use the same number of hidden neurons (num_hiddens_genotype) for every term.
        self.term_dim_map = {
            term: num_hiddens_genotype for term in term_size_map.keys()
        }

        # Create a linear layer for every term that has direct gene annotations.
        # Each such layer maps from the full gene expression vector (gene_dim) to a vector
        # with length equal to the number of genes directly annotated to that term.
        for term, gene_set in term_direct_gene_map.items():
            layer = nn.Linear(gene_dim, len(gene_set))
            setattr(self, term + "_direct_gene_layer", layer)

        # Build the ontology layers (bottom-up).
        # term_neighbor_map: for each term, store its children.
        self.term_neighbor_map = {}
        for term in dG.nodes():
            self.term_neighbor_map[term] = list(dG.neighbors(term))

        # Build term_layer_list, which is a list of lists of terms ordered from the leaves upward.
        # We make a copy of the ontology graph so we can remove nodes as we process them.
        self.term_layer_list = []
        dG_copy = dG.copy()
        while True:
            # Leaves: nodes with no outgoing edges.
            leaves = [n for n in dG_copy.nodes() if dG_copy.out_degree(n) == 0]
            if len(leaves) == 0:
                break
            self.term_layer_list.append(leaves)
            dG_copy.remove_nodes_from(leaves)
            # Note: We assume that the ontology is well-formed so that eventually only the root remains.

        # For each term in the ontology (processed bottom-up), create a module.
        # The input to each term's module is the concatenation of:
        #   - The outputs of its children modules (if any).
        #   - The output of its direct gene layer (if it has direct annotations).
        for layer in self.term_layer_list:
            for term in layer:
                input_size = 0
                # Sum up dimensions from children outputs.
                for child in self.term_neighbor_map.get(term, []):
                    # Only add child's dimension if the child has been assigned one.
                    input_size += self.term_dim_map.get(child, 0)
                # If this term has direct gene annotations, add that size.
                if term in term_direct_gene_map:
                    input_size += len(term_direct_gene_map[term])
                # Create the term module only if input_size > 0.
                if input_size > 0:
                    # Linear layer to process the concatenated inputs.
                    setattr(
                        self,
                        term + "_linear_layer",
                        nn.Linear(input_size, self.term_dim_map[term]),
                    )
                    # Batch normalization.
                    setattr(
                        self,
                        term + "_batchnorm_layer",
                        nn.BatchNorm1d(self.term_dim_map[term]),
                    )
                    # Optionally, add auxiliary layers for intermediate outputs (for interpretability or auxiliary loss).
                    setattr(
                        self,
                        term + "_aux_linear_layer1",
                        nn.Linear(self.term_dim_map[term], 1),
                    )
                    setattr(self, term + "_aux_linear_layer2", nn.Linear(1, 1))

        # Finally, create the final top layer using the root's output.
        self.final_linear_layer = nn.Linear(
            self.term_dim_map[root], self.num_hiddens_final
        )
        self.final_batchnorm_layer = nn.BatchNorm1d(self.num_hiddens_final)
        self.final_aux_linear_layer = nn.Linear(self.num_hiddens_final, 1)
        self.final_output_layer = nn.Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, gene_dim).

        Returns:
            tuple: (aux_out_map, term_out_map)
                - aux_out_map (dict): Contains auxiliary outputs for each term and the final prediction under the key 'final'.
                - term_out_map (dict): Contains the hidden representations for each term.
        """
        # x is the gene expression input.
        gene_input = x
        term_gene_out_map = {}
        # Process direct gene layers: For each term with direct gene annotations.
        for term in self.term_direct_gene_map:
            layer = getattr(self, term + "_direct_gene_layer")
            term_gene_out_map[term] = layer(gene_input)

        term_out_map = {}
        aux_out_map = {}

        # Process ontology terms in a bottom-up manner.
        for layer in self.term_layer_list:
            for term in layer:
                child_inputs = []
                # Gather outputs from children, if available.
                for child in self.term_neighbor_map.get(term, []):
                    if child in term_out_map:
                        child_inputs.append(term_out_map[child])
                # Also, include the direct gene layer output if available.
                if term in term_gene_out_map:
                    child_inputs.append(term_gene_out_map[term])
                # If there is no input (should not happen), skip this term.
                if len(child_inputs) == 0:
                    continue
                # Concatenate along feature dimension.
                combined_input = torch.cat(child_inputs, dim=1)
                linear_layer = getattr(self, term + "_linear_layer")
                bn_layer = getattr(self, term + "_batchnorm_layer")
                out = linear_layer(combined_input)
                out = torch.tanh(out)
                out = bn_layer(out)
                term_out_map[term] = out
                # Compute auxiliary output.
                aux1 = getattr(self, term + "_aux_linear_layer1")(out)
                aux1 = torch.tanh(aux1)
                aux2 = getattr(self, term + "_aux_linear_layer2")(aux1)
                aux_out_map[term] = aux2

        # Use the root term's output to compute the final prediction.
        root_output = term_out_map[self.root]
        final_out = self.final_linear_layer(root_output)
        final_out = torch.tanh(final_out)
        final_out = self.final_batchnorm_layer(final_out)
        aux_final = self.final_aux_linear_layer(final_out)
        aux_final = torch.tanh(aux_final)
        aux_final = self.final_output_layer(aux_final)
        final_prediction = torch.sigmoid(aux_final)  # Ensure output is between 0 and 1.
        aux_out_map["final"] = final_prediction
        term_out_map["final"] = final_out

        return aux_out_map, term_out_map
