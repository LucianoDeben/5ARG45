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


class genecell_nn(nn.Module):
    """
    An ontology-based neural network for gene expression data.

    This network uses a given ontology (DAG) to guide a hierarchical neural network.
    For each ontology term, a module is built that takes as input the combined signals
    from its child terms (if any) and from the genes that are directly annotated to that term.
    The network is constructed bottom-up so that the root node’s representation is used for
    the final regression prediction.
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
        self.term_dim_map = {term: num_hiddens_genotype for term in term_size_map.keys()}

        # Use ModuleDict for the direct gene layers.
        self.direct_gene_layers = nn.ModuleDict({
            term: nn.Linear(gene_dim, len(gene_set))
            for term, gene_set in term_direct_gene_map.items()
        })

        # Build the ontology layers (bottom-up).
        # Build term_neighbor_map: for each term, store its children.
        self.term_neighbor_map = {term: list(dG.neighbors(term)) for term in dG.nodes()}

        # Build term_layer_list: an ordered list of lists of terms from leaves upward.
        self.term_layer_list = []
        dG_copy = dG.copy()
        while True:
            leaves = [n for n in dG_copy.nodes() if dG_copy.out_degree(n) == 0]
            if not leaves:
                break
            self.term_layer_list.append(leaves)
            dG_copy.remove_nodes_from(leaves)

        # Create ModuleDicts for each term in the ontology (processed bottom-up).
        self.term_linear_layers = nn.ModuleDict()
        self.term_bn_layers = nn.ModuleDict()
        self.term_aux_linear_layers1 = nn.ModuleDict()
        self.term_aux_linear_layers2 = nn.ModuleDict()
        
        for layer in self.term_layer_list:
            for term in layer:
                input_size = 0
                # Sum dimensions from children outputs.
                for child in self.term_neighbor_map.get(term, []):
                    input_size += self.term_dim_map.get(child, 0)
                # If the term has direct gene annotations, add that size.
                if term in term_direct_gene_map:
                    input_size += len(term_direct_gene_map[term])
                if input_size > 0:
                    # Create and register the layers for this term.
                    self.term_linear_layers[term] = nn.Linear(input_size, self.term_dim_map[term])
                    self.term_bn_layers[term] = nn.BatchNorm1d(self.term_dim_map[term])
                    self.term_aux_linear_layers1[term] = nn.Linear(self.term_dim_map[term], 1)
                    self.term_aux_linear_layers2[term] = nn.Linear(1, 1)
                # Else: if a term receives no input, it is effectively skipped.
        
        # Final prediction head: maps the root's hidden representation to a scalar.
        self.final_linear_layer = nn.Linear(self.term_dim_map[root], self.num_hiddens_final)
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
        # Optionally, if using mixed precision, the training loop can use torch.cuda.amp.autocast.
        gene_input = x
        term_gene_out_map = {}
        # Process direct gene layers using the cached ModuleDict.
        for term, layer in self.direct_gene_layers.items():
            term_gene_out_map[term] = layer(gene_input)

        term_out_map = {}
        aux_out_map = {}

        # Process ontology terms in a bottom-up manner.
        for layer in self.term_layer_list:
            for term in layer:
                child_inputs = []
                # Gather outputs from children (if available).
                for child in self.term_neighbor_map.get(term, []):
                    if child in term_out_map:
                        child_inputs.append(term_out_map[child])
                # Also include the direct gene layer output (if available).
                if term in term_gene_out_map:
                    child_inputs.append(term_gene_out_map[term])
                # If there is no input for this term, skip its computation.
                if len(child_inputs) == 0:
                    continue
                # Concatenate the inputs along the feature dimension.
                combined_input = torch.cat(child_inputs, dim=1)
                # Look up the pre-registered layers.
                linear_layer = self.term_linear_layers[term]
                bn_layer = self.term_bn_layers[term]
                # Compute the term output.
                out = linear_layer(combined_input)
                out = torch.tanh(out)
                out = bn_layer(out)
                term_out_map[term] = out
                # Compute auxiliary output.
                aux1 = self.term_aux_linear_layers1[term](out)
                aux1 = torch.tanh(aux1)
                aux2 = self.term_aux_linear_layers2[term](aux1)
                aux_out_map[term] = aux2

        # Final prediction head using the root's output.
        root_output = term_out_map[self.root]
        final_out = self.final_linear_layer(root_output)
        final_out = torch.tanh(final_out)
        final_out = self.final_batchnorm_layer(final_out)
        aux_final = self.final_aux_linear_layer(final_out)
        aux_final = torch.tanh(aux_final)
        aux_final = self.final_output_layer(aux_final)
        final_prediction = torch.sigmoid(aux_final)  # Output between 0 and 1.
        aux_out_map["final"] = final_prediction
        term_out_map["final"] = final_out

        return aux_out_map, term_out_map
