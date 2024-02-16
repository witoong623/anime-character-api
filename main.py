from typing import Union

import numpy as np
import tensorrt as trt
from fastapi import FastAPI, UploadFile
from PIL import Image

import common


model_filename = 'one-piece-classifier-b16-fp16.engine'
input_shape = (3, 224, 224)
index_to_label = {0: 'Ace', 1: 'Akainu', 2: 'Brook', 3: 'Chopper', 4: 'Crocodile', 5: 'Franky', 6: 'Jinbei',
                  7: 'Kurohige', 8: 'Law', 9: 'Luffy', 10: 'Mihawk', 11: 'Nami', 12: 'Rayleigh', 13: 'Robin',
                  14: 'Sanji', 15: 'Shanks', 16: 'Usopp', 17: 'Zoro'}

app = FastAPI()

trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(trt_logger)
with open(model_filename, mode='rb') as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

trt_context = engine.create_execution_context()
# TODO: use set_input_shape instead
trt_context.set_binding_shape(0, (1, 3, 224, 224))


def load_normalized_test_case(test_image, pagelocked_buffer):
    ''' normalize image and copy image to host memory allocated by cudaMallocHost '''
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias (Image.LANCZOS) and transpose the image to CHW.
        c, h, w = input_shape
        image_arr = (
            np.asarray(image.resize((w, h)).convert('RGB'))
            .transpose([2, 0, 1])
            .astype(np.uint8)
            .ravel()
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0).astype(np.float32)

    normalizeed_image = normalize_image(Image.open(test_image))

    # copy to specific range of host memory
    np.copyto(pagelocked_buffer[:normalizeed_image.shape[0]], normalizeed_image)

    return test_image

@app.put('/predict_character/')
def get_prediction(image: UploadFile):
    # binding is list of pointer to memory on GPU
    # inputs, outputs are HostDeviceMem
    # stream is cuda stream created by cudaStreamCreate
    inputs, outputs, bindings, stream = common.allocate_buffers(engine, trt_context)

    load_normalized_test_case(image.file, inputs[0].host)

    [trt_outputs] = common.do_inference_v2(trt_context, bindings=bindings, inputs=inputs, outputs=outputs,
                                           stream=stream)

    character_idx = np.argmax(trt_outputs, axis=-1)

    response = {'name': index_to_label[character_idx], 'confidence': trt_outputs[character_idx].item()}

    common.free_buffers(inputs, outputs, stream)
    # after free, can't access any memory allocated by cuda.

    return response
