from typing import Union

import numpy as np
import tensorrt as trt
from fastapi import FastAPI, UploadFile
from PIL import Image

import common


model_filename = 'one-piece-classifier-b16-fp16.engine'
input_shape = (3, 224, 224)

app = FastAPI()

trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(trt_logger)
with open(model_filename, mode='rb') as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

trt_context = engine.create_execution_context()


def load_normalized_test_case(test_image, pagelocked_buffer):
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


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put('/get_characters/')
async def get_prediction(image: UploadFile):
    inputs, outputs, bindings, stream = common.allocate_buffers(engine, profile_idx=0)

    print(bindings)
    test_case = load_normalized_test_case(image.file, inputs[0].host)

    trt_outputs = common.do_inference_v2(trt_context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    return [
        {'name': 'luffy', 'conf': 0.5},
        {'name': 'nami', 'conf': 0.1},
        {'name': 'sanji', 'conf': 0.4},
        {'name': image.filename, 'conf': 0},
    ]
