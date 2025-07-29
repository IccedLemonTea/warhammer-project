import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build(socket)
# If your nn model requires 320x320 input size (BGR):
cam_out = cam.requestOutput((320, 320), dai.ImgFrame.Type.BGR888p)

nn_archive = dai.NNArchive('/home/warhammer/Downloads/yolov8ntrained.rvc2.tar.xz')
nn = pipeline.create(dai.node.NeuralNetwork).build(cam_out, nn_archive)

# Start device and get queues
device = dai.Device(pipeline)
qNNData = device.getOutputQueue(nn.out)

while True:
    inNNData: dai.NNData = qNNData.get()
    tensor = inNNData.getFirstTensor()
    assert isinstance(tensor, np.ndarray)
    print(f"Received NN data: {tensor.shape}")
