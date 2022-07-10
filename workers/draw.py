import os
import random
import importlib
import time
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
import torchvision.transforms as transforms
from PIL import Image
from config import get_config, update_config
import datasets
import models
import quant_tools
import utils
import runners
import math
import shutil
from quant_tools.blocks import (
    QuantConv2d,
    QuantLinear,
    QuantFeature,
    NAME_QBLOCK_MAPPING,
    QuantBasic,
)
from models import BLOCK_NAME_MAPPING
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

colors_per_class = {
0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
    5 : [128, 80, 128],
    6 : [87, 101, 116],
    7 : [52, 31, 151],
    8 : [0, 0, 0],
    9 : [100, 100, 255],
}

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def calibration_run(model, dataloader, device):
    model.eval()
    tmp_size = len(dataloader)
    #self._register_hook(calibration_init_param)
    iter_dataloader = iter(dataloader)
    cnt = 0
    while tmp_size > 0:
        cnt += 1
        data = next(iter_dataloader)
        if isinstance(data, (list, tuple)):
            data = data[0]
        with torch.no_grad():
            model(data.to(device))
        tmp_size -= data.shape[0]

save_name = 'simclr_4w4f.png'

def draw(args):
    config = get_config(args)
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger("Draw", config.OUTPUT)

    cudnn.benchmark = True
    fix_random_seeds( )


    # build dataloaders
    input_shape = 32
    eval_preprocess = transforms.Compose([
        transforms.Resize(int(input_shape * (8 / 7)), interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # now we forward a batch
    dataset_class = importlib.import_module(
        "datasets." + "cifar10"
    ).DATASET
    
    dataset = dataset_class(transform=eval_preprocess, mode='train')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=None, 
        batch_size=100,  # per-gpu
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )
    #dataloader = datasets.build_dataloader("train", config, eval_preprocess)

    model = models.__dict__[config.MODEL.ARCH](num_classes=config.MODEL.NUM_CLASSES)
    model.fc = nn.Identity()

    quant_flag = False
    if config.MODEL.PRETRAINED:
            if 'Q' in config.MODEL.PRETRAINED or 'pact' in config.MODEL.PRETRAINED or config.TRAIN.WARMUP_FC:
                model = quant_tools.QuantModel(model, config)
                quant_flag = True
      
            utils.train.load_ssl_checkpoint(model, path=config.MODEL.PRETRAINED, warmup_fc=config.TRAIN.WARMUP_FC)

    if not quant_flag:
        model = quant_tools.QuantModel(model, config)   
    
    model = model.to(device)
    print(model)
    model.allocate_bit(config)

    #'''
    model.reset_minmax()
    model._register_hook_update()
    calibration_run(model, dataloader, device)
    #calibration_run(model, calibration_dataloader, device)
    model._unregister_hook()
    #model.calibration(calibration_dataloader,  config.QUANT.CALIBRATION.SIZE)
    for m in model.modules():
        if isinstance(m, quant_tools.QuantLinear):
            print(m)
            if m.output_quantizer:
                m.output_quantizer.bit=0
            if m.weight_quantizer:
                m.weight_quantizer.bit=0
    #model.calibration(train_dataloader, config.QUANT.CALIBRATION.SIZE)
    model.set_quant_state(w_quant=True, a_quant=True, w_init=True, a_init=True)
    #'''

    utils.logging_information(logger, config, str(model))

    features, labels, image_paths = get_features(dataloader, model)

    tsne = TSNE(n_components=2).fit_transform(features)

    visualize_tsne(tsne, labels)

def get_features(dataloader, model):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.eval()

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for i, (images, target) in enumerate(dataloader):
        images = images.to(device)
        labels += list(target.numpy())
        #image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
        
        if i==50:
            break

    return features, labels, image_paths

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #print(labels)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label, s=2)

    # build a legend using the labels we set previously
    ax.legend(loc='best', fontsize=8)

    # finally, show the plot
    plt.savefig(save_name)
    #plt.show()


def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    #visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


