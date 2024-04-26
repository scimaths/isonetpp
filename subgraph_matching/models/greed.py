import torch
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.data import Batch

def asymm_norm(x, y, p=1, node_ins_cost=1, node_del_cost=1, edge_ins_cost=1, edge_del_cost=1):
    # return torch.norm(x - y, dim=-1, p=p)
    return (F.relu(x - y) * (node_del_cost + edge_del_cost) * 0.5).sum(dim=-1) + (F.relu(y - x) * (node_ins_cost + edge_ins_cost) * 0.5).sum(dim=-1)

class EmbedModel(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        hidden_dim,
        output_dim,
        one_hot_dim,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = one_hot_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)            

        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(
                pyg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                ))
            )

        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.pool = pyg.nn.global_add_pool

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i&1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)

        x = emb
        x = self.pool(x, g.batch)
        x = self.post(x)
        return x


class SiameseModel(torch.nn.Module):
    def __init__(
        self,
        device
    ):
        super().__init__()
        self.embed_model = None
        self.weighted = False
        self.device = device

    def forward_emb(self, gx, hx):
        raise NotImplementedError

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        gx = self.embed_model(query_batch)
        hx = self.embed_model(corpus_batch)
        return self.forward_emb(gx, hx)

    def predict_inner(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries) <= batch_size:
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            h = pyg.data.Batch.from_data_list(targets).to(self.device)
            with torch.no_grad():
                return self.forward(g, h)
        else:
            loader = pyg.data.DataLoader(list(zip(queries, targets)), batch_size, num_workers=1)
            ret = torch.empty(len(queries), device=self.device)
            for i, (g, h) in enumerate((loader, 'batches')):
                g = g.to(self.device)
                h = h.to(self.device)
                with torch.no_grad():
                    ret[i*batch_size:(i+1)*batch_size] = self.forward(g, h)
            return ret

    def predict_outer(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries)*len(targets) <= batch_size:
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            h = pyg.data.Batch.from_data_list(targets).to(self.device)
            gx = self.embed_model(g)
            hx = self.embed_model(h)
            with torch.no_grad():
                return self.forward_emb(gx[:,None,:], hx)
        else:
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            gx = self.embed_model(g)
            loader = pyg.data.DataLoader(targets, batch_size//len(queries), num_workers=1)
            ret = torch.empty(len(queries), len(targets), device=self.device)
            for i, h in enumerate((loader, 'batches')):
                h = h.to(self.device)
                hx = self.embed_model(h)
                with torch.no_grad():
                    ret[:,i*loader.batch_size:(i+1)*loader.batch_size] = self.forward_emb(gx[:,None,:], hx)
            return ret

class Greed(SiameseModel):
    def __init__(
        self,
        node_ins_cost,
        node_del_cost,
        edge_ins_cost,
        edge_del_cost,
        output_mode,
        n_layers,
        hidden_dim,
        output_dim,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        device,
    ):
        super().__init__(device)
        self.embed_model = EmbedModel(
            n_layers,
            hidden_dim,
            output_dim,
            input_dim
        )
        self.output_mode = output_mode
        self.node_ins_cost = node_ins_cost
        self.node_del_cost = node_del_cost
        self.edge_ins_cost = edge_ins_cost
        self.edge_del_cost = edge_del_cost

    def forward_emb(self, x, y):
        return asymm_norm(
            x, y, int(self.output_mode[-1]), self.node_ins_cost, self.node_del_cost, self.edge_ins_cost, self.edge_del_cost
        )
