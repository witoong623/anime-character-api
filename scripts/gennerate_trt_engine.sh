trtexec --onnx=one-piece-classifier.onnx \
	--minShapes=images:1x3x224x224 --optShapes=images:16x3x224x224 \
	--maxShapes=images:16x3x224x224 --fp16 \
	--saveEngine=one-piece-classifier-b16-fp16.engine
