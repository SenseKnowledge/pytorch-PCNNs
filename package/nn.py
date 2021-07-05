# -*- coding: utf-8 -*-
import torch


class PiecewiseMaxPool(torch.nn.Module):
    """Applies a Piecewise Max Pooling over an input signal composed of several input planes.
    """

    def __init__(self, num_pieces: int):
        """
        num_pieces: int
            number of pieces
        """
        super(PiecewiseMaxPool, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x: torch.Tensor, mask: torch.LongTensor):
        """

        Parameters
        ----------
        x: torch.Tensor, [batch_size, length, feature_dim]
        mask: torch.Tensor, [batch_size, length]
            mask for different pieces, ignore 0

        Returns
        -------
        [batch_size, num_pieces, feature_dim]
        """

        xs = []
        for i in range(1, self.num_pieces + 1):
            mask_piece = (1. - torch.eq(mask, i).float()) * -100.
            x_piece = torch.clamp(x + mask_piece.unsqueeze(-1), min=0.)
            x_piece = torch.max(x_piece, 1).values  # (batch_size, feature_dim)
            xs.append(x_piece)

        x = torch.stack(xs, 1)  # (batch, num_pieces, feature_dim)

        return x


if __name__ == '__main__':
    pool = PiecewiseMaxPool(3)
    mask = torch.LongTensor([[0, 0, 1, 1, 0, 0, 2, 2, 2]])
    x = torch.Tensor([[[1, 1, 1, 1, 1], [1, 10, 1, 1, 1],
                       [20, 2, 2, 2, 2], [2, 20, 2, 2, 2], [2, 2, 20, 2, 2], [2, 2, 2, 20, 2],
                       [30, 3, 3, 3, 3], [3, 3, 30, 3, 3], [3, 30, 3, 3, 30]]])

    print(pool(x, mask))
