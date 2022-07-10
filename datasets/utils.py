import cv2
import numpy as np


def workaround_torch_size_bug(size):
    assert len(size) == 2, "only support (h, w)"
    if size[0] == size[1]:
        return size[0]
    return size


def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)

    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image

    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)

    assert img is not None, "failed to decode"
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img
