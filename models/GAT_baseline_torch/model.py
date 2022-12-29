import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter, Linear

from weights_init import glorot, zeros
from torch_geometric.utils import softmax
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.nn import GATConv,GCNConv

class PreNormException(Exception):
    pass

class PreNormLayer(nn.Module):
    """
    Our pre-normalization layer, whose purpose is to normalize an input layer
    to zero mean and unit variance to speed-up and stabilize GCN training. The
    layer's parameters are aimed to be computed during the pre-training phase.
    """
    def __init__(self, n_units, shift=True, scale=True):
        super(PreNormLayer, self).__init__()
        assert shift or scale

        if shift:
            self.register_buffer(f"shift", torch.zeros((n_units,), dtype=torch.float32))
        else:
            self.shift = None

        if scale:
            self.register_buffer(f"scale", torch.ones((n_units,), dtype=torch.float32))
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input):
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input = input+ self.shift

        if self.scale is not None:
            input = input * self.scale

        return input

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.shape[-1]}."

        input = input.reshape([-1, self.n_units])
        sample_avg = torch.mean(input, dim=0)
        sample_var = torch.mean((input - sample_avg) ** 2, dim=0)
        sample_count = input.numel() / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """

        assert self.count > 0
        if self.shift is not None:
            self.shift = - self.avg

        if self.scale is not None:
            self.var = torch.where(torch.eq(self.var, 0.0), torch.ones_like(self.var), self.var) # NaN check trick
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        
        return self.scale, self.shift

class BaseModel(torch.nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                self.initializer(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train(self, state):
        with torch.no_grad():
            try:
                self.forward(state)
                return False
            except PreNormException:
                return True

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) \
                    and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))



class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation, heads=2, center_left=True,):
        super().__init__()
        print('heads', heads)
        self.conv1 = GATConv(in_channels, out_channels,heads=heads, edge_dim =out_channels, add_self_loops=False)#, fill_value =1.0)#add_self_loops =False)
        self.linear_transform = nn.Linear(out_channels*heads, out_channels)
        self.activation= activation
        
        self.post_concat = torch.nn.Linear(out_channels*2, out_features=out_channels)
        
        
        self.feature_module_final = nn.Sequential(
            # PreNormLayer(1, shift=False),  # normalize after summation trick
            self.activation,
            nn.Linear(out_channels, out_channels, bias=True)
        )

    def forward(self, inputs):
        left_features, edge_index, edge_features, right_features, scatter_out_size = inputs
      
        x=(right_features,left_features)
        x, attention_weights = self.conv1(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
        x = x.relu()
        x = self.linear_transform(x)#.relu()
        x = self.feature_module_final(x)
        left_features_updated = torch.cat((left_features,x), dim=-1)
        
        left_features_updated=self.post_concat(left_features_updated)

        return left_features_updated, attention_weights

class GATPolicy(BaseModel):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self):
        super(GATPolicy, self).__init__()

        self.emb_size =32
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19
        
        print('self.emb_size',self.emb_size)

        self.activation = nn.ReLU()
        self.initializer = lambda x: torch.nn.init.orthogonal_(x, gain=1)
        self.dict_norm_task = None

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
            nn.Linear(self.cons_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation
        )
        #
        # EDGE EMBEDDING
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.edge_nfeats, self.emb_size, bias=True),
            
            
        )
        #
        # # VARIABLE_EMBEDDING
        self.var_embedding = nn.Sequential(
            nn.Linear(self.var_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation
        )

        # GRAPH CONVOLUTIONS
        self.conv_v_to_c = GATNet(self.emb_size,self.emb_size, self.emb_size, activation = self.activation)#, right_to_left=True)
        self.conv_c_to_v = GATNet(self.emb_size, self.emb_size, self.emb_size, activation = self.activation)
       
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, 1, bias=False)
        )
       

        self.post_conv_module = nn.Sequential(
        )

        self.initialize_parameters()

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)

        output = torch.split(
            tensor=output,
            split_size_or_sections=n_vars_per_sample.tolist(),
            dim=1,
        )

        output = torch.cat([
            F.pad(x,
                pad=[0, n_vars_max - x.shape[1], 0, 0],
                mode='constant',
                value=pad_value)
            for x in output
        ], dim=0)

        return output

    def forward(self, inputs, return_attention_weights=None):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.
        """
        

        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        n_cons_total = torch.sum(n_cons_per_sample)
        n_vars_total = torch.sum(n_vars_per_sample)

        
        edge_indices_rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        

            
        constraint_features, attention_weights1 = self.conv_v_to_c((
            constraint_features, edge_indices_rev, edge_features, variable_features, n_cons_total))
        
      
        
        
        constraint_features = self.activation(constraint_features)
        
        variable_features, attention_weights2= self.conv_c_to_v((
            variable_features, edge_indices, edge_features, constraint_features, n_vars_total))
        
        variable_features = self.activation(variable_features)
       

        # OUTPUT
        output = self.output_module(variable_features)
        output = torch.reshape(output, [1, -1])

        if(return_attention_weights is  None):
            return variable_features, output
        else:
            return variable_features, output, attention_weights1,attention_weights2
