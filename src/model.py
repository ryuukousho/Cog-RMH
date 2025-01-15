from abc import ABC, abstractmethod
from layers import *
from utils import givens_rotations, givens_reflection, mobius_add, expmap0, project, hyp_distance_multi_c, logmap0, operations


class KGModel(nn.Module, ABC):
    def __init__(self, sizes, rank, dropout, gamma, bias, init_size, use_cuda=False):
        super(KGModel, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.rel = nn.Embedding(sizes[1], rank)
        self.wordrel = nn.Embedding(sizes[3], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1))
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1))

    def forward(self, queries, ent_emb, eval_mode=False, rel_emb=None, c=None):
        lhs_e, lhs_biases = self.get_queries(queries, ent_emb)
        rhs_e, rhs_biases = self.get_rhs(queries, ent_emb, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)
        factors = self.get_factors(queries, ent_emb)
        return predictions, factors

    @abstractmethod
    def get_queries(self, queries, ent_emb):
        pass

    @abstractmethod
    def get_rhs(self, queries, ent_emb, eval_mode):
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries, ent_emb):
        head_e = ent_emb[queries[:, 0]]
        rel_e = self.rel(queries[:, 1])
        rhs_e = ent_emb[queries[:, 2]]
        return head_e, rel_e, rhs_e


class BaseH(KGModel):
    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.bias, args.init_size)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank))
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank)) - 1.0
        self.wordrel.weight.data = self.init_size * torch.randn((self.sizes[3], 2 * self.rank))
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1))
        else:
            c_init = torch.ones((1, 1))
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, ent_emb=None, eval_mode=False):
        if eval_mode:
            return ent_emb, self.bt.weight
        else:
            return ent_emb[queries[:, 2]], self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    def get_c(self):
        if self.multi_c:
            return self.c
        else:
            return self.c.repeat(self.sizes[1], 1)


class AttH(BaseH):
    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank)) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank))
        self.act_att = nn.Softmax(dim=1)
        self.scale = nn.Parameter(torch.Tensor([1. / np.sqrt(self.rank)]), requires_grad=False)

    def get_queries(self, queries, ent_emb):
        head = ent_emb[queries[:, 0]]
        c_p = self.get_c()
        c = F.softplus(c_p[queries[:, 1]])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act_att(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class RMVH(AttH):
    def __init__(self, args):
        super(RMVH, self).__init__(args)
        self.device = args.device
        self.n_layers = args.n_layers
        self.history_len = args.history_len
        self.en_dropout = args.dropout
        self.de_dropout = args.de_dropout
        self.up_dropout = args.up_dropout
        self.model_name = args.model

        self.init_ent_emb = nn.Embedding(self.sizes[0], self.rank)
        self.init_ent_emb.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))
        self.word_emb_temp = nn.Embedding((self.sizes[2]-self.sizes[0]), self.rank)
        self.word_emb_temp.weight.data = self.init_size * torch.randn(((self.sizes[2]-self.sizes[0]), self.rank))

        self.entity_cell2 = nn.GRUCell(self.rank, self.rank)
        self.entity_cell3 = nn.GRUCell(self.rank, self.rank)
        self.h = None
        self.static_h = None
        assert args.n_layers > 0
        self.update = TF(self.rank, args.sizes[0], args.use_time, self.history_len, self.init_ent_emb, args.n_head,
                         args.up_dropout, args.layer_norm, args.double_precision, self.init_size)
        self.layers_e = nn.ModuleList()
        self.layers_s = nn.ModuleList()
        self.static_layers = nn.ModuleList()
        self.build_layers(args)
        if self.model_name == 'RMVH':
            self.s_hp = nn.Parameter(torch.Tensor([args.s_hp]), requires_grad=False)
            self.s_delta_ind = args.s_delta_ind
            if args.s_hp < 0:
                if args.s_delta_ind:
                    self.delta_l = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
                    self.delta_r = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
                else:
                    self.delta = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
            self.score_comb = operations[args.s_comb]
            self.score_softmax = args.s_softmax
            self.s_dropout = args.s_dropout
            self.reason_dropout = args.reason_dropout

    def build_layers(self, args):
        for i in range(self.n_layers):
            self.layers_e.append(RGCNLayer(self.rank, self.sizes[1], self.rel, self.en_dropout, args.en_loop, self.init_size, args.en_bias))
            self.layers_s.append(RGCNLayer(self.rank, self.sizes[1], self.rel, self.en_dropout, args.en_loop, self.init_size, args.en_bias))
            self.static_layers.append(RGCNLayer(self.rank, self.sizes[3], self.wordrel, self.en_dropout, args.en_loop, self.init_size, args.en_bias))

    def evolve(self, g_list, static_graph):
        static_g = static_graph.to(self.device)
        self.h = self.init_ent_emb.weight.clone()
        word_emb_h = self.word_emb_temp.weight.clone()
        word_emb = torch.cat((self.h, word_emb_h), dim=0)
        self.static_h = self.static_forward(static_g, word_emb)
        self.h_init = self.init_ent_emb.weight.clone()
        g_list_temp = g_list
        g_list.reverse()
        g_list.extend(g_list_temp)
        evolve_embs_e = []
        evolve_embs_s = []
        for idx in range(len(g_list)):
            if g_list[idx].device == "cpu":
                g = g_list[idx].to(self.device)
            else:
                g = g_list[idx]
            hidden_e = self.snap_forward_e(g, self.h)
            hidden_e = self.entity_cell2(hidden_e, self.h_init)
            hidden_s = self.snap_forward_s(g, self.static_h)
            hidden_s = self.entity_cell3(hidden_s, self.h_init)
            evolve_embs_e.append(hidden_e)
            evolve_embs_s.append(hidden_s)
        evolve_snap_e = self.update(evolve_embs_e)[-1]
        evolve_snap_s = self.update(evolve_embs_s)[-1]
        output_emb = self.update([evolve_snap_e, evolve_snap_s])[-1]
        return output_emb

    def snap_forward_e(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id]
        for i, layer in enumerate(self.layers_e):
            layer(g)
        return g.ndata.pop('h')

    def snap_forward_s(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id]
        for i, layer in enumerate(self.layers_s):
            layer(g)
        return g.ndata.pop('h')

    def static_forward(self, static_g, word_emb):
        node_id = static_g.ndata['id'].squeeze()
        static_g.ndata['h'] = word_emb[node_id]
        for i, layer in enumerate(self.static_layers):
            layer(static_g)
        return static_g.ndata.pop('h')[:self.sizes[0], :]

    def comb_score(self, queries, old_score, new_score, act=torch.sigmoid):
        if self.score_softmax:
            old_score = torch.softmax(old_score, 1, old_score.dtype)
            new_score = torch.softmax(new_score, 1, new_score.dtype)
        if self.s_hp[0] < 0:
            if self.s_delta_ind:
                w1 = self.delta_l[queries[:, 0]]
                w2 = self.delta_r[queries[:, 2]]
            else:
                w1 = self.delta[queries[:, 0]]
                w2 = self.delta[queries[:, 2]]
            if act:
                w1 = act(w1)
                w2 = act(w2)
            w = self.score_comb(w1, w2)
            w = F.dropout(w, self.up_dropout, training=self.training)
        else:
            w = self.s_hp.repeat(queries.shape[0], 1)
        score = w * new_score + (1 - w) * old_score
        return score

    def reason(self, queries, ent_emb, eval_mode=False, epoch=1000, rel_emb=None, c=None):
        new_factors, old_factors = None, None
        if self.model_name == 'RMVH' and self.s_hp != 0:
            new_ent_emb = F.dropout(ent_emb, self.reason_dropout, training=self.training)
            init_ent_emb = self.init_ent_emb.weight
            new_score, new_factors = self.forward(queries, new_ent_emb, eval_mode=eval_mode)
            old_score, old_factors = self.forward(queries, init_ent_emb, eval_mode=eval_mode)
            score = self.comb_score(queries, old_score, new_score)
        else:
            score, factor = self.forward(queries, ent_emb, eval_mode=eval_mode)
        return score, (old_factors, new_factors)