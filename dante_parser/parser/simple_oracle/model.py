import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import itemgetter
from typing import Dict, List

from dante_parser.parser.simple_oracle.archybrid import Configuration

global SHIFT, LEFT_ARC, RIGHT_ARC, SWAP
SHIFT, LEFT_ARC, RIGHT_ARC, SWAP = 0, 1, 2, 3


class TransitionBasedDependencyParsing(nn.Module):
    def __init__(
        self,
        pos_dic: Dict[str, int],
        emb_dim: int,
        pos_dim: int,
        hidden_dim: int,
        hidden_dim2: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.pos_dic = pos_dic  # Ditionary of my pos-tags
        self.emb_dim = emb_dim  # Size of lstm embedding
        self.pos_dim = pos_dim  # Size of the pos-tags dimension
        self.hidden_dim = hidden_dim  # Size of the hidden dim
        self.hidden_dim2 = hidden_dim2

        self.lstm = nn.LSTM(
            emb_dim + pos_dim,  # Essa concatenação é curiosa, qual o motivo dela?
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_dim * 2 * 4, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, 4)

    def forward(self, sentence: List[str]):
        """
        Network forward with batch_size of 1.

        Parameters
        ----------
        sentence: List[str]
            List of tokens.
        """
        # batch size = 1
        x = []
        for word in sentence:
            # use one-hot to decode pos tags
            pos_vec = [0.0] * self.pos_dim
            # TODO: adapt to TokenList from conllu package.
            if word["upos"] in self.pos_dic:
                pos_id = self.pos_dic[word["upos"]]
            else:  # Assign UNK to pos-tag.
                pos_id = self.pos_dim - 1
            pos_vec[pos_id] = 1.0
            # It reads a pre-trained embed and concatenates the pos-tag one-hot
            # TODO: read embeddings for each token from a pre-trained model.
            import pdb;pdb.set_trace()
            for i in range(len(word["emb"])):
                word["emb"][i] = float(word["emb"][i])
            x.append(word["emb"] + pos_vec)
            # x.append(pos_vec)
        # IMPORTANT: The input of this network is the concatenation of a embedding representation
        # plus the pos-tag one-hot representation
        x = torch.tensor(x).cuda()
        # x = [sent len, emb dim + pos dim]
        x = x.unsqueeze(1)
        # x = [sent len, batch size, emb dim + pos dim]

        output, (hidden, cell) = self.lstm(x)
        # output = [sent len, batch size, hid dim*num directions]
        output = output.squeeze(1)
        # output = [sent len, hid dim*num directions]

        # model for training
        if self.training:
            # Initialize an arc-hybrid system
            config = Configuration(sentence, sufficient_info=True)

            action_cnt = 0
            mlosses = []
            mloss = 0.0
            err_cnt = 0
            while not config.is_empty():
                # Extract features
                # b0
                if config.buffer[0] == 0:
                    feature_b0 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_b0 = output[config.buffer[0] - 1]
                # s0
                if config.stack.__len__() < 1 or config.stack[-1] == 0:
                    feature_s0 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s0 = output[config.stack[-1] - 1]
                # s1
                if config.stack.__len__() < 2 or config.stack[-2] == 0:
                    feature_s1 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s1 = output[config.stack[-2] - 1]
                # s2
                if config.stack.__len__() < 3 or config.stack[-3] == 0:
                    feature_s2 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s2 = output[config.stack[-3] - 1]
                # Concat 4 features
                features = torch.cat(
                    (feature_s2, feature_s1, feature_s0, feature_b0)
                ).cuda()
                features = features.unsqueeze(0)
                # features = [batch, hidden_dim*2*4]
                out = F.leaky_relu(self.fc1(features))
                # out = [batch, hid_dim2]
                out = F.softmax(self.fc2(out))
                # out = [batch, 4]
                out = out.squeeze(0)
                # out = [4]

                # TODO: Calculate the costs
                costs, shift_case = config.calculate_cost()

                # Choose the most possible valid action and wrong action
                # Each item is represented as (Action, Prob(tensor), Prob)
                bestValid = []
                bestWrong = []
                for i, prob in zip(range(4), out):
                    if costs[i] == 0:
                        bestValid.append((i, prob, prob.item()))
                    else:
                        bestWrong.append((i, prob, prob.item()))
                bestValid = max(bestValid, key=itemgetter(2))
                bestWrong = max(bestWrong, key=itemgetter(2))

                #####################
                # We can try 'aggresive exploration'
                # in 4.1 "Simple and Accurate Dependency Parsing"
                #####################

                # Updates for the dynamic oracle
                config.oracle_update(bestValid[0], shift_case)
                # Apply the best action
                if bestValid[0] == SHIFT or bestValid[0] == SWAP:
                    config.apply_transition(bestValid[0])
                else:
                    config.apply_transition((bestValid[0], None))

                # Updates magin loss
                if bestValid[2] < bestWrong[2] + 1.0:
                    mloss += 1.0 + bestWrong[2] - bestValid[2]
                    mlosses.append(bestWrong[1] - bestValid[1])

                action_cnt += 1

            return mlosses, mloss

        # model for testing
        else:
            # Initialize an arc-hybrid system for testing
            config = Configuration(sentence, sufficient_info=True)

            action_cnt = 0
            err_cnt = 0
            while not config.is_empty():
                # Extract features
                # b0
                if config.buffer[0] == 0:
                    feature_b0 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_b0 = output[config.buffer[0] - 1]
                # s0
                if config.stack.__len__() < 1 or config.stack[-1] == 0:
                    feature_s0 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s0 = output[config.stack[-1] - 1]
                # s1
                if config.stack.__len__() < 2 or config.stack[-2] == 0:
                    feature_s1 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s1 = output[config.stack[-2] - 1]
                # s2
                if config.stack.__len__() < 3 or config.stack[-3] == 0:
                    feature_s2 = torch.zeros(self.hidden_dim * 2).cuda()
                else:
                    feature_s2 = output[config.stack[-3] - 1]
                # Concat 4 features
                features = torch.cat(
                    (feature_s2, feature_s1, feature_s0, feature_b0)
                ).cuda()
                features = features.unsqueeze(0)
                # features = [batch, hidden_dim*2*4]
                out = F.leaky_relu(self.fc1(features))
                # out = [batch, hid_dim2]
                out = F.softmax(self.fc2(out))
                # out = [batch, 4]
                out = out.squeeze(0)
                # out = [4]

                # Choose the most possible valid action and wrong action
                # Each item is represented as (Action, Prob(tensor), Prob)
                bestValid = []
                for i, prob in zip(range(4), out):
                    if config.transition_admissible(i):
                        bestValid.append((i, prob, prob.item()))
                bestValid = max(bestValid, key=itemgetter(2))

                if bestValid[0] == LEFT_ARC or bestValid[0] == RIGHT_ARC:
                    child = config.stack[-1]
                    pred_father = (
                        config.buffer[0]
                        if bestValid[0] == LEFT_ARC
                        else config.stack[-2]
                    )

                # Apply the best action
                if bestValid[0] == SHIFT or bestValid[0] == SWAP:
                    config.apply_transition(bestValid[0])
                else:
                    config.apply_transition((bestValid[0], None))

                if bestValid[0] == LEFT_ARC or bestValid[0] == RIGHT_ARC:
                    if config.father[child] != pred_father:
                        err_cnt += 1

                action_cnt += 1

            return err_cnt, config
