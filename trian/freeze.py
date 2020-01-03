import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import paddle
import paddle.fluid as fluid
import reader
#from mobilenet_ssd import build_mobilenet_ssd
from shufflenetv2_ssd import build_ssd
from utility import add_arguments, print_arguments, check_cuda

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('model_dir',        str,   'model_snet/best_model',     "The model path.")
add_arg('model_save_dir',   str,   'model_snet/300_300_infer_model',     "The model path.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")



def infer(args, model_dir):
    image_shape = [3, args.resize_h, args.resize_w]
    num_classes = 81

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    locs, confs, box, box_var = build_ssd(image, num_classes,
                                                    image_shape)
    #nmsed_out = fluid.layers.detection_output(
    #    locs, confs, box, box_var, nms_threshold=args.nms_threshold)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    if model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(model_dir, var.name))
        fluid.io.load_vars(exe, model_dir, predicate=if_exist)
        print("load model succeed")
    
    bboxes = paddle.fluid.layers.box_coder(prior_box=box,prior_box_var=box_var,target_box=locs,code_type='decode_center_size')
    scores = fluid.layers.softmax(input=confs)
    scores = fluid.layers.transpose(scores, perm=[0, 2, 1])

    print("boxes.shape = ",bboxes.shape)
    print("scores.shape = ",scores.shape)

    print("start transform model")
    fluid.io.save_inference_model(args.model_save_dir,['image'],[bboxes,scores], exe)
    print("transform model succeed")

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    check_cuda(args.use_gpu)

    infer(args,model_dir=args.model_dir)
