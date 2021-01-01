#!/usr/bin/env python3

import os
import numpy as np
import torch
from engine import train_one_epoch, evaluate
import utils as u
from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH = "../exercise_ws/src/obj_det/include/model"


class DuckieDataset(Dataset):
    def __init__(self, root, domain="sim", transforms=None):
        self.root = root
        self.npz = list(sorted([os.path.join(root, x) for x in os.listdir(root)]))
        self.transforms = transforms
        self.domain = domain
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.bbox = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        path = self.npz[idx]
        data = np.load(path)#, allow_pickle=True, encoding='latin1'
        img = Image.fromarray(data[f"arr_{0}"]).convert("RGB")
        boxes = data[f"arr_{1}"]
        classes = data[f"arr_{2}"]
        

        # get bounding box coordinates for each mask
        num_objs = len(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(classes, dtype=torch.int64)

        image_id = torch.tensor([idx])
        if num_objs>0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.as_tensor([[0,0,1,1]], dtype=torch.float32)
            labels = torch.as_tensor([0], dtype=torch.int64)
            area = torch.as_tensor([1], dtype=torch.int64)
        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        img = transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.npz)


def get_transform(train):
    # might do something later
    if train:
        transform = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.2,
                hue=0.2,
            )
        return transform
    else:
        return None


    
# let's train it for 10 epochs
def save(model, optimizer, output_path, step, save_n_epochs):
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "object_detect_latest.pth"

        # Construct relevant state dicts / optims:

        save_dict = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "step": step,
        }
        if step % save_n_epochs == 0:
            torch.save(
                save_dict, save_dir / f"object_detect_epoch_{step}_ckpt.pth"
            )
            print("saved model in " + str(save_path))

        torch.save(save_dict, save_path)
        print("saved model in " + str(save_path))


def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!
    # use our dataset and defined transformations
    dataset = DuckieDataset(
        "/miniscratch/tengmeli/duckietown/sim_dataset_object/dataset_duckietown_vf/", domain="sim", transforms=get_transform(train=True)
    )
    dataset_test = DuckieDataset(
       "/miniscratch/tengmeli/duckietown/sim_dataset_object/dataset_duckietown_vf/", domain="sim", transforms=get_transform(train=False)
    )

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=u.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=u.collate_fn,
    )
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 5

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)
    num_epochs = 200

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        print(epoch)
        save(model, optimizer, "./models/", epoch, save_n_epochs=2)
        #didn't use
        #evaluate(model, data_loader_test, device, epoch + 1)

if __name__ == "__main__":
    main()
