import torch
import os
import argparse
import shutil
import utils
import numpy as np
import random
import torchvision.transforms.functional as transforms_F

from torch import optim
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, models
from datetime import datetime
from torchvision.datasets import ImageFolder
from losses import TripletLossHuman
from model import FTModel
from tqdm import tqdm

current_time = datetime.now().strftime("%d_%m_%Y-%H_%M")

parser = argparse.ArgumentParser(description='Material Similarity Training')
parser.add_argument('--train-dir',
                    metavar='DIR', help='path to dataset',
                    default='data/split_dataset')
parser.add_argument('--test-dir',
                    metavar='DIR', help='path to dataset',
                    default='data/havran1_ennis_298x298_LDR')
parser.add_argument('-j', '--workers',
                    default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs',
                    default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--adjust-epoch',
                    nargs='+', default=[10, 15, 20],
                    type=int, help='milestones to adjust the learning rate')
parser.add_argument('--num-classes', default=100, type=int,
                    help='number of classes in the problem')
parser.add_argument('--emb-size',
                    default=128, type=int, help='size of the embedding')
parser.add_argument('-b', '--batch-size',
                    default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay',
                    default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--betas',
                    nargs='+', default=[0.9, 0.999], type=float,
                    help='beta values for ADAM')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='momentum in the SGD')
parser.add_argument('--margin',
                    default=0.3, type=float,
                    help='triplet loss margin')
parser.add_argument('--checkpoint-folder',
                    default='./checkpoints',
                    type=str, help='folder to store the trained models')
parser.add_argument('--model-name',
                    default='resnet_similarity', type=str,
                    help='name given to the model')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',
                    dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=2851, type=int,
                    help='seed for initializing training. ')


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/single.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, img):
        size = np.random.randint(self.low, self.high)
        return transforms_F.resize(img, size, self.interpolation)


def train_model(loader, epoch):
    def update_progress_bar(progress_bar, losses):
        description = '[' + str(epoch) + '-train]'
        description += ' Triplet loss: '
        description += '%.4f/ %.4f (AVG)' % (losses.val, losses.avg)
        progress_bar.set_description(description)

    global model
    global criterion
    global optimizer

    # keep track of the loss value
    losses = AverageMeter()

    progress_bar = tqdm(loader, total=len(loader))
    for imgs, targets in progress_bar:
        with torch.set_grad_enabled(True):
            imgs = imgs.to(device, dtype)
            targets = targets.to(device, dtype)

            # forward through the model and compute error
            _, embeddings = model(imgs)
            loss = criterion(embeddings, targets)
            losses.update(loss.item(), imgs.size(0))

            # compute gradient and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_progress_bar(progress_bar, losses)

    return losses.avg


def evaluate_model(mturk_images):
    global model
    global criterion
    global optimizer

    with torch.set_grad_enabled(False):
        # get current agreement with users answers
        current_agreement_train = criterion.get_majority_accuracy(
            mturk_images=mturk_images,
            model=model,
            train=True,
            unit_norm=True
        )
        current_agreement = criterion.get_majority_accuracy(
            mturk_images=mturk_images,
            model=model,
            train=False,
            unit_norm=True
        )

    tqdm.write('[Train]Current agreement %.4f (Best agreement %.4f)' %
               (current_agreement_train, best_agreement))
    tqdm.write('[Test]Current agreement %.4f (Best agreement %.4f)' %
               (current_agreement, best_agreement))

    return current_agreement


def get_transforms():
    # set image transforms
    trf_train = transforms.Compose([
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.CenterCrop(size=384),
        RandomResize(low=256, high=384),
        transforms.RandomCrop(size=224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trf_test = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
    ])
    return trf_train, trf_test


def get_dataloaders(trf_train):
    global args

    def _init_loader_(worker_id):
        np.random.seed(args.seed + worker_id)

    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': True,
        'worker_init_fn': _init_loader_,
    }

    loader_train = DataLoader(
        dataset=ImageFolder(
            root=os.path.join(args.train_dir, 'train'),
            transform=trf_train,
        ),
        shuffle=True,
        drop_last=True,
        **loader_args
    )
    loader_val = DataLoader(
        dataset=ImageFolder(
            root=os.path.join(args.train_dir, 'val'),
            transform=trf_train,
        ),
        shuffle=True,
        **loader_args
    )

    return loader_train, loader_val


def save_checkpoint(state, is_best, folder, model_name='checkpoint', ):
    """
    if the current state is the best it saves the pytorch model
    in folder with name filename
    """
    path = os.path.join(folder, model_name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'model')

    torch.save(state, path + '.pth.tar')
    if is_best:
        shutil.copyfile(path + '.pth.tar', path + '_best.pth.tar')


if __name__ == '__main__':

    # get input arguments
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device and dtype
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # # comment this if we want reproducibility
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = True

        # # this might affect performance but allows reproducibility
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    # define dataset
    trf_train, trf_test = get_transforms()
    loader_train, loader_val = get_dataloaders(trf_train)
    mturk_images, _ = utils.load_imgs(args.test_dir, trf_test)

    # create model
    model = FTModel(
        models.resnet34(pretrained=True),
        layers_to_remove=1,
        num_features=args.emb_size,
        num_classes=args.num_classes,
    )
    model = model.to(device, dtype)

    # define loss function
    criterion = TripletLossHuman(
        margin=args.margin,
        unit_norm=True,
        device=device,
        seed=args.seed
    )

    # define optimizer
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     betas=args.betas,
    #     weight_decay=args.weight_decay,
    #     lr=args.lr,
    #     amsgrad=True
    # )
    optimizer = optim.SGD(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True
    )

    # define LR scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.adjust_epoch,
        gamma=0.1,
    )

    # set a high value for the error
    best_agreement = 0

    if args.evaluate:
        # evaluation step
        model = model.eval()
        evaluate_model()

    else:
        # start training and evaluation loop
        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            # train step
            model = model.train()
            train_model(loader_train, epoch)
            lr_scheduler.step()

            # evaluation step
            model = model.eval()
            current_agreement = evaluate_model(mturk_images)

            # checkpoint model if it is the best
            is_best = current_agreement > best_agreement
            best_agreement = max(current_agreement, best_agreement)
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_agreement': best_agreement,
                    'optimizer': optimizer.state_dict(),
                },
                is_best, folder=args.checkpoint_folder,
                model_name=args.model_name + '-' + current_time
            )
