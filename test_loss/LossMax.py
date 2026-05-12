import torch
import torch.nn as nn

from Conv import Conv
from torch.nn.functional import binary_cross_entropy_with_logits


class LossMax():
    def __init__(self):
        self.convs = nn.ModuleList([
            Conv(1, 1, 3),
            Conv(1, 1, 7),
            Conv(1, 1, 9),
            # Conv(1, 1, 15),
            # Conv(1, 1, 19),
            # Conv(1, 1, 23),
            # Conv(1, 1, 27),
            # Conv(1, 1, 31)
        ])
        self.optims = [torch.optim.Adam(conv.parameters(), lr=0.01) for conv in self.convs]

    def find_max_conv_block(self, pred, target, step, loss_fn):
        for convI in range(len(self.convs)):
            for i in range(step):
                self.optims[convI].zero_grad()
                # Detach pred to avoid gradient accumulation across iterations
                pred_input = pred.detach() if i > 0 else pred
                pred_new = self.convs[convI](pred_input)
                loss = loss_fn(pred_new, target)
                loss.backward(retain_graph=(i < step - 1))
                self.optims[convI].step()

    def apply_pred(self, pred):
        for optm in self.optims:
            optm.zero_grad()
        for conv in self.convs:
            pred = conv(pred)

        return pred

    def to_device(self, device):
        for conv in self.convs:
            conv.to(device)
























        
