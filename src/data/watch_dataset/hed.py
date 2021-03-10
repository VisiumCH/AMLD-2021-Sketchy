import cv2 as cv
import os


# Extracted from https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class HED():
    ''' Class for performing inference using the HED model '''

    def __init__(self, path_to_pretrained_model_dir):

        weights_path, def_file_path = self.__get_model_files(path_to_pretrained_model_dir)

        cv.dnn_registerLayer('Crop', CropLayer)

        self.model = cv.dnn.readNet(cv.samples.findFile(def_file_path), cv.samples.findFile(weights_path))

    def __get_model_files(self, path_to_pretrained_model_dir):
        ''' Returns the path to the model weights and definition file '''
        weights_path = os.path.join(path_to_pretrained_model_dir, 'hed_pretrained_bsds.caffemodel')
        definition_file_path = os.path.join(path_to_pretrained_model_dir, 'deploy.prototxt')

        return weights_path, definition_file_path

    def predict(self, input_image):
        ''' Returns the HED output for the input image '''
        inp = cv.dnn.blobFromImage(input_image, scalefactor=1.0, size=(500, 500),
                                   mean=(104.00698793, 116.66876762, 122.67891434),
                                   swapRB=False, crop=False)
        self.model.setInput(inp)
        out = self.model.forward()
        out = out[0, 0]
        out = cv.resize(out, (input_image.shape[1], input_image.shape[0]))

        return out