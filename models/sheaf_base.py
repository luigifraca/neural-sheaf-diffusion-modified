# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from lib.edge_coupling import validate_edge_index


class SheafDiffusion(nn.Module):
    """Base class for sheaf diffusion models."""

    def __init__(self, edge_index, args):
        super(SheafDiffusion, self).__init__()

        assert args['d'] > 0
        self.d = args['d']
        validate_edge_index(edge_index, num_nodes=args['graph_size'])
        self.edge_index = edge_index
        self.add_lp = args['add_lp']
        self.add_hp = args['add_hp']

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.hidden_dim = args['hidden_channels'] * self.final_d
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.normalised = args['normalised']
        self.deg_normalised = args['deg_normalised']
        self.nonlinear = not args['linear']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        self.left_weights = args['left_weights']
        self.right_weights = args['right_weights']
        self.sparse_learner = args['sparse_learner']
        self.use_act = args['use_act']
        self.input_dim = args['input_dim']
        self.hidden_channels = args['hidden_channels']
        self.output_dim = args['output_dim']
        self.layers = args['layers']
        self.sheaf_act = args['sheaf_act']
        self.second_linear = args['second_linear']
        self.orth_trans = args['orth']
        self.use_edge_weights = args['edge_weights']
        self.t = args['max_t']
        self.time_range = torch.tensor([0.0, self.t], device=self.device)
        self.laplacian_builder = None

    def update_edge_index(self, edge_index):
        validate_edge_index(edge_index, num_nodes=self.graph_size)
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def _reset_node_representations(self, x=None):
        self._last_node_representations = {}
        if x is not None:
            self._store_node_representation("input", x)

    def _store_node_representation(self, name, x):
        x_detached = x.detach()
        if x_detached.dim() == 1:
            x_detached = x_detached.unsqueeze(-1)
        if x_detached.size(0) == self.graph_size * self.final_d:
            x_detached = x_detached.reshape(self.graph_size, -1)
        elif x_detached.size(0) == self.graph_size:
            x_detached = x_detached.reshape(self.graph_size, -1)
        self._last_node_representations[name] = x_detached.cpu()

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others
