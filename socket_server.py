from viewer.predictor import RetinaNetPredictor
import socket
from PIL import Image
import numpy as np
import skimage
import json


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')


model_path = r'./models/tiff_retinanet_199.pt'
predictor = RetinaNetPredictor()
if predictor.model_load(model_path):
    print('starting server...')

sock = socket.socket()
sock.bind(('', 27015))
print('server started, waiting for connection...')
sock.listen(1)

while True:
    conn, addr = sock.accept()
    print('connected: ', addr)

    expected = 640000
    actual = 0
    data = b''
    while actual < expected:
        # print(actual, expected)
        more_data = conn.recv(expected - actual)
        data += more_data
        actual += len(more_data)
    print(len(data), ' bytes received')
    img = Image.frombytes("L", (800, 800), data)
    img = np.array(img)
    img = skimage.color.grey2rgb(img)
    bboxes = predictor.get_bboxes(img)

    print(bboxes, bboxes.shape)

    # bboxes = bboxes.tobytes()
    # num = len(bboxes)
    # print(num)
    msg = json.dumps(bboxes.tolist()).encode('utf-8')
    print(msg)
    conn.send(msg)
        # conn.send(data.upper())
    
    conn.close()
