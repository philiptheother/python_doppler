import torch
import torch.nn as nn

from diffuser.gnns.graphs.RGCN import RGCNRootShared

class RGCN_MLP_LN(nn.Module):
    def __init__(self,
                 state_dim,
                 num_inputs,
                 hidden_size=256,
                 ln=False,
                 ltl_embed_input_dim=22,
                 ltl_embed_output_dim=32,
                 ltl_embed_hidden_dim=32,
                 ltl_embed_num_layers=8,
                 output=print):
        super(RGCN_MLP_LN, self).__init__()

        self.state_dim = state_dim
        self.ltl_embed_output_dim = ltl_embed_output_dim
        self.gnn = RGCNRootShared(
            ltl_embed_input_dim,
            ltl_embed_output_dim,
            hidden_dim=ltl_embed_hidden_dim,
            num_layers=ltl_embed_num_layers
        )

        n_param = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        output(f"GNN Number of parameters: {n_param}")

        num_inputs = num_inputs + ltl_embed_output_dim

        if ln:
            self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, 1))
        else:
            self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, 1))
            
        # Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.train()

    def forward(self, x, formulas):
        formula_embeds = self.gnn(formulas)
        embeds = torch.cat([x, formula_embeds], dim=-1)
        return self.mlp(embeds)
