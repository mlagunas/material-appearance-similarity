import numpy as np
import scipy.io
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import transform

np.random.seed(123)
torch.manual_seed(123)


def _imscatter(x, y, image, color=None, ax=None, zoom=1.):
    """ Auxiliary function to plot an image in the location [x, y]
        image should be an np.array in the form H*W*3 for RGB
    """
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
        size = min(image.shape[0], image.shape[1])
        image = transform.resize(image[:size, :size], (256, 256))
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        edgecolor = dict(boxstyle='round,pad=0.05',
                         edgecolor=color, lw=4) \
            if color is not None else None
        ab = AnnotationBbox(im, (x0, y0),
                            xycoords='data',
                            frameon=False,
                            bboxprops=edgecolor,
                            )
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


if __name__ == '__main__':
    embs_path = 'data/embs_havran_ennis.mat'
    do_unit_norm = False

    mat_file = scipy.io.loadmat(embs_path)
    embs = torch.tensor(mat_file['embs'])

    if do_unit_norm:
        embs /= embs.norm(p=2, dim=1, keepdim=True)
        embs = embs.numpy()

    img_paths = [str(elem).strip() for elem in mat_file['img_paths']]

    # get umap from the embeddings
    umap_fit = umap.UMAP(n_neighbors=30,
                         init='spectral',
                         min_dist=4,
                         spread=8,
                         metric='l1')
    umap_emb = umap_fit.fit_transform(embs)

    # plot each point of the umap as its corresponding image
    fig = plt.figure()
    ax = fig.gca()
    for i, point in enumerate(umap_emb):
        _imscatter(point[0], point[1], img_paths[i], zoom=0.12, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()
