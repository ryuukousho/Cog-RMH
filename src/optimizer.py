import numpy as np
import torch
from tqdm import tqdm
import random
from utils import build_sub_graph
from utils import get_ranking
from torch import nn


class KGOptimizer(object):
    def __init__(self, model, optimizer, ft_epochs, norm_weight, valid_freq, history_len, multi_step, topk, batch_size, neg_sample_size,
                 double_neg=False, metrics='raw', use_cuda=False, dropout=0., verbose=True, grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.grad_norm = grad_norm
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.module.sizes[0]
        self.n_relations = model.module.sizes[1]
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.valid_freq = valid_freq
        self.multi_step = multi_step
        self.topk = topk
        self.history_len = history_len
        self.metrics = metrics
        self.dropout = dropout
        self.ft_epochs = ft_epochs
        self.norm_weight = norm_weight

    def calculate_loss(self, out_g, ent_emb, epoch=-1):
        loss = torch.zeros(1).cuda().to(self.device) if self.use_cuda else torch.zeros(1)
        mean_score = None
        score, factors = self.model.module.reason(out_g, ent_emb, eval_mode=True, epoch=epoch)
        truth = out_g[:, 2]
        loss += self.loss_fn(score, truth)
        return loss, mean_score

    def epoch(self, train_list, static_graph, epoch=-1):
        losses = []
        idx = [_ for _ in range(len(train_list))]
        random.shuffle(idx)
        score_of_snap = np.zeros(len(train_list))
        for train_sample_num in tqdm(idx):
            if train_sample_num == 0:
                continue
            output = train_list[train_sample_num]
            if train_sample_num - self.history_len < 0:
                input_list = train_list[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - self.history_len:train_sample_num]
            history_g_list = [build_sub_graph(self.n_entities, self.n_relations, snap, self.use_cuda, self.device, self.dropout) for snap in input_list]
            output = torch.from_numpy(output).long().cuda().to(self.device) if self.use_cuda else torch.from_numpy(output).long()
            evolve_ent_emb = self.model.module.evolve(history_g_list, static_graph)
            loss, mean_score = self.calculate_loss(output, evolve_ent_emb, epoch=epoch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())
        return np.mean(losses), score_of_snap


    def evaluate(self, history_list, test_list, static_graph, filtered_ans_list, filters, valid_mode=False, epoch=-1, multi_step_test=False, topk_test=0):
        valid_losses = []
        valid_loss = None
        ranks = []
        filter_ranks = []
        input_list = [snap for snap in history_list[-self.history_len:]]
        with torch.no_grad():
            for time_idx, test_snap in enumerate(tqdm(test_list)):
                history_g_list = [build_sub_graph(self.n_entities, self.n_relations, g, self.use_cuda, self.device) for g in input_list]
                test_triples = torch.LongTensor(test_snap).cuda().to(self.device) if self.use_cuda else torch.LongTensor(test_snap)
                evolve_ent_emb = self.model.module.evolve(history_g_list, static_graph)
                if valid_mode:
                    loss, mean_score = self.calculate_loss(test_triples, evolve_ent_emb, epoch=epoch)
                    valid_losses.append(loss.item())
                if (epoch + 1) % self.valid_freq == 0 or not valid_mode:
                    score, _ = self.model.module.reason(test_triples, evolve_ent_emb, eval_mode=True, epoch=epoch)
                    _, _, rank, filter_rank = get_ranking(test_triples, score, filtered_ans_list[time_idx], filters, self.metrics, batch_size=self.batch_size)
                    ranks.append(rank)
                    filter_ranks.append(filter_rank)
                    input_list.pop(0)
                    input_list.append(test_snap)
            if valid_losses:
                valid_loss = np.mean(valid_losses)
            if ranks:
                ranks = torch.cat(ranks)
            if filter_ranks:
                filter_ranks = torch.cat(filter_ranks)
            return valid_loss, ranks, filter_ranks