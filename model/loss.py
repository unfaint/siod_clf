import torch


class RegClsLoss(torch.nn.Module):
    """
    Combination of L2 and cross entropy loss, inspired by
    https://arxiv.org/abs/1506.01497
    """

    def __init__(self):
        super(RegClsLoss, self).__init__()
        self.reg_criterion = torch.nn.MSELoss(reduction= 'none')
        self.clf_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, outputs_l, outputs_c, labels):
        batch_size = labels.shape[0]

        reg_loss = self.reg_criterion(outputs_l, labels[:,:2])
        clf_loss = self.clf_criterion(outputs_c, labels[:,2].long())

        # leave loss for positive samples, zero everything else

        reg_loss = labels[:,2].view(batch_size, -1) * reg_loss
        reg_loss = torch.mean(reg_loss)

        loss = reg_loss + clf_loss

        return loss