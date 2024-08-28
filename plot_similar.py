import os
import random
import shutil

import scipy.io
import torch
from matplotlib import pyplot as plt

import utils


def plot_tensor(tensor, title="", folder=None):
    """ plots a torch tensor as an image. 'title' will be the image title
        and 'folder' the folder where it will be stored
    """
    # move to numpy and change shape from CxWxH to WxHxC
    np_array = tensor.cpu().numpy().transpose((1, 2, 0))

    plt.imshow(np_array.squeeze(), cmap='Greys_r')
    plt.title(title)

    # if a folder is given store in the folder otherwise plot
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, title + '.png')
        plt.savefig(path)
    else:
        plt.show()


def copy_k_closer(reference_ix, k, dist, image_paths, folder='./'):
    os.makedirs(folder, exist_ok=True)

    # get image extension
    ext = image_paths[0].split('.')[-1]

    # copy reference image
    shutil.copy(img_paths[reference_ix], os.path.join(folder, 'reference.' + ext))

    # obtain closer images to the reference according to the features
    min_dist, min_idx = torch.topk(dist[reference_ix], k=k, largest=False)

    # plot closer images to the reference
    for i in range(k):
        dist, idx = min_dist[i], min_idx[i]
        shutil.copy(img_paths[idx],
                    os.path.join(folder, '%d-dist:%2.2f.%s' % (i + 1, dist, ext)))


if __name__ == '__main__':
    embs_path = 'data/embs_havran_ennis.mat'  # /mat file with the embeddings
    n_close_elems = 5  # number of close elements to find
    reference_img = 'data/havran1_ennis_298x298_LDR/aluminium.jpg'
    do_unit_norm = True

    # load embeddings
    print('loading embedding')
    mat_file = scipy.io.loadmat(embs_path)
    embs = torch.tensor(mat_file['embs'])

    if do_unit_norm:
        embs /= embs.norm(p=2, dim=1, keepdim=True)

    img_paths = [str(elem).strip() for elem in mat_file['img_paths']]

    # get pairwise distances between features
    print('getting distances')
    all_images_dist = utils.pairwise_dist(embs)
    # make diagonal a big number
    all_images_dist[range(len(all_images_dist)), range(len(all_images_dist))] = 9999

    # plot all the images
    try:
        reference_ix = img_paths.index(reference_img)
    except ValueError:
        reference_ix = random.randint(0, len(img_paths))
        reference_img = img_paths[reference_ix]
        print('reference image not found. Randomly selected %s as reference img'
              % reference_img)

    # create folder with the name of the reference file
    out_folder = 'data/%s' % (reference_img.split('/')[-1].split('.')[0])
    copy_k_closer(reference_ix, k=n_close_elems,
                  dist=all_images_dist, image_paths=img_paths, folder=out_folder)
    print('output is stored in %s' % out_folder)
