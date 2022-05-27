from flask import Flask, request
import os
from connection import s3_connection, s3_put_object
from config import BUCKET_NAME
import time
import module.core.test

from dotmap import DotMap

import module.background_color as bc
import numpy as np
import cv2

app = Flask(__name__)

# @app.route("/api",methods=["GET"])
# def test(): 
#    return {
#    "message": "sucess"
#  }

@app.route("/api/image", methods=["POST"])
def react_to_flask():
    # 딥러닝
    myTest = module.core.test
    args = {
        'cuda': True,
        'dataset_choice': "SAMPLES",
        'model_path': "module/models/pretrained/p3mnet_pretrained_on_p3m10k.pth",
        'test_choice': "HYBRID"
    }

    m = DotMap(args)
    myTest.load_model_and_deploy(m)

    input_img = cv2.imread('static/color/hello.png', cv2.IMREAD_COLOR)
    mask_img = cv2.imread('static/alpha/hello.png', cv2.IMREAD_COLOR)

    input_img = input_img.reshape(-1, input_img.shape[0], input_img.shape[1], input_img.shape[2])
    mask_img = mask_img.reshape(-1, mask_img.shape[0], mask_img.shape[1], mask_img.shape[2])

    img = bc.inference(input_img, mask_img)
    print(img.shape)

    #사진 리턴
    return {
    "success": True, "img_url": "http://localhost:5000/static/inference.jpg"
}


if __name__ == '__main__':
    print('his')

    app.run(debug=True)

    print('hi')
