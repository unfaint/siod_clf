import torch
import numpy as np


def detect_objects(inputs, model, rf_centers, c_threshold= 0.55,
                   dist_threshold= 18, n= 3):
    outputs_l, outputs_c = model(inputs)
    outputs_c_clone = outputs_c.clone()

    coords = None

    locations = outputs_l * 10 + rf_centers
    outputs_c_long = outputs_c.squeeze().view(2, -1)
    outputs_c_long[1,:][outputs_c_long[1,:] < c_threshold] = 0
    locations = locations.squeeze().view(2, -1)

    switch = True

    i = 0

    if torch.sum(outputs_c_long[1,:]) < 1:
        switch = False

    while switch:
        max_c, max_ix = torch.max(outputs_c_long[1,:].view(1, -1), dim= 1)

        if coords is None:
            coords = locations[:,max_ix].cpu().data.numpy()
        else:
            coords = np.concatenate([coords, locations[:,max_ix].cpu().data.numpy()], axis= 1)

        l2 = torch.norm(locations - locations[:,max_ix], p= 2, dim= 0)

        outputs_c_long[1,l2 < dist_threshold] = 0

        i += 1

        if torch.sum(outputs_c_long[1,:]) < 0.1:
            switch = False

        if n != 0:
            if i == n:
                switch = False

    return coords, outputs_c_clone
