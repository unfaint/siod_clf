import math

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out

def get_receptive_fields(convnet, imsize):
    layerInfos = []
    currentLayer = [imsize, 1, 1, 0.5]
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)

    return layerInfos


def receptive_field_center(layer_names, layerInfos, idx_x=0, idx_y=0, layer_name='conv4_2'):
    layer_idx = layer_names.index(layer_name)
    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert (idx_x < n)
    assert (idx_y < n)

    return int(start + idx_x * j), int(start + idx_y * j)
