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

        outputs_l = outputs_l.view(batch_size, -1)
        outputs_c = outputs_c.view(batch_size, -1)

        reg_loss = self.reg_criterion(outputs_l, labels[:,:2])
        clf_loss = self.clf_criterion(outputs_c, labels[:,2].long())

        # leave loss for positive samples, zero everything else

        reg_loss = labels[:,2].view(batch_size, -1) * reg_loss
        reg_loss = torch.mean(reg_loss)

        loss = reg_loss + clf_loss

        return loss

class MinDistLoss(torch.nn.Module):


    def __init__(self, rf_centers, th_c= 0.6, th_dist= 12):
        """
        :param rf_centers: CUDA
        """
        super(MinDistLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.rf_centers = rf_centers
        self.th_c = th_c
        self.th_dist = th_dist

    def forward(self, outputs_l, outputs_c, labels):

        IMG_H = 193

        locations = outputs_l * 10 + self.rf_centers
        locations = locations.view(locations.shape[0], locations.shape[1], -1) # TODO replace with unfold

        x_dist = torch.log(torch.exp(labels[0, :, 0].float().cuda().unsqueeze(1) / IMG_H).matmul(
            1 / torch.exp(locations[0, 0, :].unsqueeze(1).t() / IMG_H)))
        y_dist = torch.log(torch.exp(labels[0, :, 1].float().cuda().unsqueeze(1) / IMG_H).matmul(
            1 / torch.exp(locations[0, 1, :].unsqueeze(1).t() / IMG_H)))
        xy_dist = torch.pow(x_dist, 2) + torch.pow(y_dist, 2)
        xy_dist = xy_dist * IMG_H # TODO replace hardcoded dims with variables

        N_sq = outputs_l.size(2)
        dist_fold = torch.nn.functional.fold(torch.unsqueeze(xy_dist, 0), N_sq, (1, 1))
        min_dist = torch.min(dist_fold, dim=1)[0]


        # TP
        tp_mask = (outputs_c[:, 1] > self.th_c) & (min_dist < self.th_dist) # TODO refactor to get rid of repetitive code

        tp0 = outputs_c[:, 0][tp_mask]
        tp_n = len(tp0)

        # FP
        fp_mask = (outputs_c[:, 1] > self.th_c) & (min_dist > self.th_dist)

        fp0 = outputs_c[:, 0][fp_mask]
        fp_n = len(fp0)


        # TN
        tn_mask = (outputs_c[:, 0] > self.th_c) & (min_dist > self.th_dist)

        tn0 = outputs_c[:, 0][tn_mask]
        tn_n = len(tn0)


        # FN
        fn_mask = (outputs_c[:, 0] > self.th_c) & (min_dist < self.th_dist)

        fn0 = outputs_c[:, 0][fn_mask]
        fn_n = len(fn0)

        min_n = min([tp_n, fp_n, tn_n, fn_n])
        min_n = 10 if min_n == 0 else min_n

        # TP

        if tp_n > 0:
            tp0 = tp0.unsqueeze(0)
            tp1 = outputs_c[:, 1][tp_mask].unsqueeze(0)
            tp = torch.cat([tp0, tp1], dim=0).unsqueeze(0)
            tp = tp[:,:,:min_n]

        # FP

        if fp_n > 0:
            fp0 = fp0.unsqueeze(0)
            fp1 = outputs_c[:, 1][fp_mask].unsqueeze(0)
            fp = torch.cat([fp0, fp1], dim=0).unsqueeze(0)
            fp = fp[:,:,:min_n]

        # TN

        if tn_n > 0:
            tn0 = tn0.unsqueeze(0)
            tn1 = outputs_c[:, 1][tn_mask].unsqueeze(0)
            tn = torch.cat([tn0, tn1], dim=0).unsqueeze(0)
            tn = tn[:,:,:min_n]

        # FN

        if fn_n > 0:
            fn0 = fn0.unsqueeze(0)
            fn1 = outputs_c[:, 1][fn_mask].unsqueeze(0)
            fn = torch.cat([fn0, fn1]).unsqueeze(0)
            fn = fn[:,:,:min_n]



        tp_loss = self.criterion(tp, torch.ones((tp.size(0), tp.size(2))).long().cuda()) if tp_n > 0 else 0

        fp_loss = self.criterion(fp, torch.zeros((fp.size(0), fp.size(2))).long().cuda()) if fp_n > 0 else 0

        tn_loss = self.criterion(tn, torch.zeros((tn.size(0), tn.size(2))).long().cuda()) if tn_n > 0 else 0

        fn_loss = self.criterion(fn, torch.ones((fn.size(0), fn.size(2))).long().cuda()) if fn_n > 0 else 0

        loss = tp_loss + fp_loss + tn_loss + fn_loss

        return loss