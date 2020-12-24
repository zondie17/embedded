import cv2
import threading
import numpy as np
from model import siamese

from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

input_shape = (45, 60, 3)
w = input_shape[1]
h = input_shape[0]
model = siamese(input_shape)
weight_path = "mobilenet_weight.h5"
model.load_weights(weight_path)

template_size = (140, 105)
top_k = 3
pyr_count = 2

orig_img = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
colored_img = cv2.imread("1.jpg")
pd_img = orig_img.copy()
for i in range(pyr_count):
    pd_img = cv2.pyrDown(pd_img)
cap = cv2.VideoCapture("http://localhost:8080/?action=stream")

show_img = orig_img.copy()

# http server handler
class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global show_img
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    img_str = cv2.imencode('.jpg', show_img)[1]
                    self.wfile.write("--jpgboundary".encode('utf-8'))
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',len(img_str))
                    self.end_headers()
                    self.wfile.write(img_str)
                except KeyboardInterrupt:
                    break
                    return

        if self.path == '/':
            self.path = "index.html"
            super().do_GET()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

def server_start():
    server = ThreadingHTTPServer(('0.0.0.0', 8081), MyHandler)
    print("running...")
    server.serve_forever()

def get_frame():
    global ret, frame
    while True:
        ret, frame = cap.read()

# get frames from camera in another thread
frame = 0
ret = False
t = threading.Thread(target=get_frame)
t.setDaemon(True)
t.start()

# run server in another thread
server_t = threading.Thread(target=server_start)
server_t.setDaemon(True)
server_t.start()

while True:
    if not ret:
        continue
    template = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.resize(template, template_size)
    for i in range(pyr_count):
        template = cv2.pyrDown(template)

    img = pd_img.copy()

    img = img.astype(np.float32)
    template = template.astype(np.float32)
    # naive template matching
    res_mat = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    # if top_k > 1, use siamese network to compare candidates
    if top_k > 1:
        template = cv2.resize(template, (w, h))
        uridx = np.unravel_index(res_mat.argsort(axis=None)[::-1][:top_k], res_mat.shape)
        cand_idx = np.stack(uridx, axis=-1)
        template_img = np.repeat(template[..., np.newaxis], 3, -1)
        template_img = template_img.astype(np.float64) / 255
        predict_data = [np.zeros((top_k, h, w, 3)) for i in range(2)]
        for i in range(top_k):
            xy = cand_idx[i]
            cand_img = img[xy[0]:xy[0]+h, xy[1]:xy[1]+w]
            cand_img = np.repeat(cand_img[..., np.newaxis], 3, -1)
            cand_img = cv2.resize(cand_img.astype(np.float64) / 255, (w, h))
            predict_data[0][i, ...] = template_img
            predict_data[1][i, ...] = cand_img.astype(np.float64) / 255
        probs = model.predict(predict_data)
        max_loc = cand_idx[probs.argmax()]
        print(probs.max())
        tl = (max_loc[1], max_loc[0])
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_mat)
        tl = max_loc
    img = img.astype(np.uint8)

    img = colored_img.copy()
    # compute original coordinate
    ratio_x = tl[0] / pd_img.shape[1]
    ratio_y = tl[1] / pd_img.shape[0]
    tl = (int(img.shape[1] * ratio_x), int(img.shape[0] * ratio_y))

    # show predicted position on original image
    line_color = (255, 255, 255)
    line_thickness = 1
    center_x = tl[0] + int(template_size[0] / 2)
    center_y = tl[1] + int(template_size[1] / 2)
    cv2.line(img, (0, center_y), (tl[0], center_y), line_color, line_thickness)
    cv2.line(img, (tl[0] + template_size[0], center_y), (img.shape[1], center_y), line_color, line_thickness)
    cv2.line(img, (center_x, 0), (center_x, tl[1]), line_color, line_thickness)
    cv2.line(img, (center_x, tl[1] + template_size[1]), (center_x, img.shape[0]), line_color, line_thickness)
    cv2.rectangle(img, tl, (tl[0] + template_size[0], tl[1] + template_size[1]), (255, 255, 255), 1)
    print(tl)

    show_img = img.copy()
