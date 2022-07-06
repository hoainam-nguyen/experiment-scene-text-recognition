import torch
from torch import nn


class LossFunction(nn.Module):
    def __init__(self, converter, gamma = 2.):
        super(LossFunction, self).__init__()

        self.gamma = gamma
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.CC_loss = ClusterCharLoss(converter)

    def forward(self, preds, targets):
        preds, targets = preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1)
        cost_CELoss = self.CE_loss(preds, targets)
        cost_CCLoss = self.CC_loss(preds, targets)
        cost_FCLoss = ((1 - torch.exp(-cost_CCLoss))**self.gamma) * cost_CCLoss
        total_loss = cost_CELoss + cost_FCLoss
        return total_loss


class ClusterCharLoss(nn.Module):
    def __init__(self, converter):
        super().__init__()
        self.define_cluster = [
                                "0QODo",
                                "123456789",
                                "aA",
                                "bB",
                                "cC",
                                "defgh",
                                "ijIJ",
                                "kK",
                                "uUVvyWw",
                                "mnMN",
                                "pq",
                                "rstxz",
                                "EFGHLPRSTXYZ",
                                "!|[\]/{}()l",
                                "\"'*+,-.:;<=>^_`~",
                                "#$%&€£¥°₹?@",
                            ]

        self.converter = converter
        self.map_cluster = {}
        self.get_cluster()

    def get_cluster(self):
        for i, chars in enumerate(self.define_cluster):
            for c in chars:
                self.map_cluster[self.converter.dict[c]] = i
        for c in self.converter.character:
            if self.converter.dict[c] not in self.map_cluster:
                self.map_cluster[self.converter.dict[c]] = -1


    def compute(self, preds, targets):
        total_score = []
        for p, q in zip(preds, targets):
            if not (p == q):
                if not (self.map_cluster[int(p)] == self.map_cluster[int(q)]):
                    penalty = 1
                else:
                    penalty = 0.5
            else:
                penalty = 0
            total_score.append(penalty)
        total_score = torch.Tensor(total_score)
        return total_score.mean()

    def forward(self, preds, targets):
        _, preds = preds.max(1)
        return self.compute(preds, targets)
        
