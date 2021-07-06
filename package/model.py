# -*- coding: utf-8 -*-
from package.nn import PiecewiseMaxPool
from transformers import BertModel
import torch


class PCNNsConfig(object):
    """Config class for
    """

    def __init__(self, num_pieces, num_relations, max_length, position_embedding_dim,
                 word_embedding_dim, kernel_sizes, hidden_dims, dropout_rate):
        self.position_embedding_dim = position_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.num_relations = num_relations
        self.max_length = max_length
        self.kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (tuple, list)) else (kernel_sizes,)
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, (tuple, list)) else (hidden_dims,)
        self.num_pieces = num_pieces
        self.dropout_rate = dropout_rate


class PCNNs(torch.nn.Module):
    """ Piecewise Convolutional Neural Networks
    ref 'Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks. EMNLP. 2015'
    """

    def __init__(self, lms: BertModel, cfg: PCNNsConfig):
        super(PCNNs, self).__init__()

        self.convs = torch.nn.Sequential()

        hiddens = [cfg.word_embedding_dim + 2 * cfg.position_embedding_dim] + list(cfg.hidden_dims)
        for i in range(len(hiddens) - 1):
            self.convs.add_module("conv_%s" % i, torch.nn.Conv1d(hiddens[i], hiddens[i + 1],
                                                                 cfg.kernel_sizes[i], padding=cfg.kernel_sizes[i] // 2))
            if i != len(hiddens) - 2:
                self.convs.add_module("relu_%s" % i, torch.nn.ReLU())

        self.pool = PiecewiseMaxPool(cfg.num_pieces)
        self.dropout = torch.nn.Dropout(cfg.dropout_rate)
        self.linear = torch.nn.Linear(cfg.hidden_dims[-1] * cfg.num_pieces, cfg.num_relations)

        self.pos_h_embedding = torch.nn.Embedding(2 * cfg.max_length, cfg.position_embedding_dim, padding_idx=0)
        self.pos_t_embedding = torch.nn.Embedding(2 * cfg.max_length, cfg.position_embedding_dim, padding_idx=0)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.lms = lms

    def reset_parameters(self):
        """Init all weights
        """
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def multi_instance_learning_loss(self, bags):
        """Bag-based Distant Supervision"""

        batch_logits = []
        for token, pos_h, pos_t, mask, label in zip(bags['bags_of_token_ids'], bags['bags_of_pos_h_ids'],
                                                    bags['bags_of_pos_t_ids'],
                                                    bags['bags_of_mask'], bags['label']):

            logits = self.forward(token, pos_h, pos_t, mask)

            # sample
            idx = logits[:, int(label)].argmax()
            batch_logits.append(logits[idx])

        batch_logits = torch.cat(batch_logits)
        return self.criterion(batch_logits, bags['label'])

    def fast_multi_instance_learning_loss(self, bags):
        """Fast version Bag-based Distant Supervision"""

        token, pos_h, pos_t, mask = (bags['bags_of_token_ids'],
                                     bags['bags_of_pos_h_ids'], bags['bags_of_pos_t_ids'],
                                     bags['bags_of_mask'])

        logits = self.forward(token, pos_h, pos_t, mask)

        sample_logits = []
        for label, (lhs, rhs) in zip(bags['label'], bags['scope']):
            # sample
            idx = logits[lhs:rhs][:, label].argmax()
            sample_logits.append(logits[lhs + idx])
        sample_logits = torch.stack(sample_logits)

        return self.criterion(sample_logits, bags['label'])

    def forward(self, input_ids: torch.LongTensor, pos_h: torch.LongTensor, pos_t: torch.LongTensor,
                mask: torch.LongTensor):
        """

        Parameters
        ----------
        input_ids: torch.LongTensor, [batch_size, length, feature_dim]
        pos_h: torch.LongTensor, postion ids, [batch_size, length, feature_dim]
        pos_t: torch.LongTensor, postion ids, [batch_size, length, feature_dim]
        mask: torch.LongTensor, [batch_size, length]
            mask for different pieces, ignore 0

        Returns
        -------
        torch.Tensor, (batch_size, num_relations)

        """
        x = self.lms(input_ids=input_ids, attention_mask=torch.gt(mask, 0).long())[0]

        x = torch.cat([x,
                       self.pos_h_embedding(pos_h),
                       self.pos_t_embedding(pos_t)], -1)

        x = x.transpose(1, 2)  # [batch_size, feature_dim, length]
        x = self.convs(x)
        x = x.transpose(1, 2)  # [batch_size, length, feature_dim]
        x = self.pool(x, mask).view(x.size(0), -1)  # [batch_size, num_pieces * feature_dim]
        x = self.dropout(x)
        x = self.linear(x)
        return x
