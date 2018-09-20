import torch

import torch.nn.functional


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

class MinDistLoss(torch.nn.Module):


    def __init__(self, rf_centers):
        """


        :param rf_centers: CUDA
        """
        super(MinDistLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.rf_centers = rf_centers

    def forward(self, outputs_l, outputs_c, labels):
        locations = outputs_l * 10 + self.rf_centers
        x_dist = torch.log(torch.exp(labels[0, :, 0].float().cuda().unsqueeze(1) / 257).matmul(
            1 / torch.exp(locations[0, 0, :].unsqueeze(1).t() / 257)))
        y_dist = torch.log(torch.exp(labels[0, :, 1].float().cuda().unsqueeze(1) / 257).matmul(
            1 / torch.exp(locations[0, 1, :].unsqueeze(1).t() / 257)))
        xy_dist = torch.pow(x_dist, 2) + torch.pow(y_dist, 2)
        xy_dist = xy_dist * 257 # TODO replace hardcoded dims with variables

        N_sq = outputs_l.size(2)
        dist_fold = torch.nn.functional.fold(torch.unsqueeze(xy_dist, 0), N_sq, (1, 1))
        min_dist = torch.min(dist_fold, dim=1)[0]

        th_c = 0.5
        th_dist = 12

        # TP
        tp_mask = (outputs_c[:, 1] > th_c) & (min_dist < th_dist) # TODO refactor to get rid of repetitive code

        tp0 = outputs_c[:, 0][tp_mask]
        tp_n = len(tp0)
        if tp_n > 0:
            tp0 = tp0.unsqueeze(0)
            tp1 = outputs_c[:, 1][tp_mask].unsqueeze(0)
            tp = torch.cat([tp0, tp1], dim=0).unsqueeze(0)

        # FP
        fp_mask = (outputs_c[:, 1] > th_c) & (min_dist > th_dist)

        fp0 = outputs_c[:, 0][fp_mask]
        fp_n = len(fp0)
        if fp_n > 0:
            fp0 = fp0.unsqueeze(0)
            fp1 = outputs_c[:, 1][fp_mask].unsqueeze(0)
            fp = torch.cat([fp0, fp1], dim=0).unsqueeze(0)

        # TN
        tn_mask = (outputs_c[:, 0] > th_c) & (min_dist > th_dist)

        tn0 = outputs_c[:, 0][tn_mask]
        tn_n = len(tn0)
        if tn_n > 0:
            tn0 = tn0.unsqueeze(0)
            tn1 = outputs_c[:, 1][tn_mask].unsqueeze(0)
            tn = torch.cat([tn0, tn1], dim=0).unsqueeze(0)

        # FN
        fn_mask = (outputs_c[:, 0] > th_c) & (min_dist < th_dist)

        fn0 = outputs_c[:, 0][fn_mask]
        fn_n = len(fn0)
        if fn_n > 0:
            fn0 = fn0.unsqueeze(0)
            fn1 = outputs_c[:, 1][fn_mask].unsqueeze(0)
            fn = torch.cat([fn0, fn1]).unsqueeze(0)

        eps = 0.001

        tp_w = 10 / (tp_n + eps) if tp_n > 0 else 0

        fp_w = 10 / (fp_n + eps) if fp_n > 0 else 0

        tn_w = 10 / (tn_n + eps) if tn_n > 0 else 0

        fn_w = 10 / (fn_n + eps) if fn_n > 0 else 0

        tp_loss = self.criterion(tp, torch.ones((tp.size(0), tp.size(2))).long().cuda()) if tp_n > 0 else 0

        fp_loss = self.criterion(fp, torch.zeros((fp.size(0), fp.size(2))).long().cuda()) if fp_n > 0 else 0

        tn_loss = self.criterion(tn, torch.zeros((tn.size(0), tn.size(2))).long().cuda()) if tn_n > 0 else 0

        fn_loss = self.criterion(fn, torch.ones((fn.size(0), fn.size(2))).long().cuda()) if fn_n > 0 else 0

        loss = tp_w * tp_loss + fp_w * fp_loss + tn_w * tn_loss + fn_w * fn_loss

        return loss