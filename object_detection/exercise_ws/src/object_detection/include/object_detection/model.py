import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import numpy as np


class NoGPUAvailable(Exception):
    print("NO GPU")
    pass


class Wrapper:
    def __init__(self, model_file):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        # TODO If no GPU is available, raise the NoGPUAvailable exception

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise (NoGPUAvailable)
            # self.device = torch.device("cpu")
        # our dataset has two classes only - background and person
        num_classes = 5
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device)["model"]
        )
        self.model.eval()

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        # or simply pipe the whole batch to the model instead of using a loop!
        image = torch.from_numpy(np.transpose(batch_or_image, (2, 0, 1))).to(
            self.device
        )
        image = image / 255
        try:
            box, label, score = self.model(
                image.unsqueeze(0)
            )  # TODO you probably need to send the image to a tensor, etc.
        except:
            box = []
            label = []
            score = []
        boxes.append(box)
        labels.append(label)
        scores.append(score)

        return boxes, labels, scores


class Model:  # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        # TODO Instantiate your weights etc here!
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False
        )

        # get number of input features for the classifier
        self.in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    # TODO add your own functions if need be!
