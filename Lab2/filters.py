import numpy as np
from numpy.lib.stride_tricks import as_strided


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    img_i = i + k - Hk // 2
                    img_j = j + l - Wk // 2
                    if 0 <= img_i < Hi and 0 <= img_j < Wi:
                        out[i, j] += image[img_i, img_j] * kernel[Hk - k - 1, Wk - l - 1]
    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    H, W = image.shape
    out = np.zeros_like(np.empty((H + 2 * pad_height, W + 2 * pad_width)))
    out[pad_height: H + pad_height, pad_width: W + pad_width] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    Hb, Wb = Hk // 2, Wk // 2
    padded_img = zero_pad(image, Hb, Wb)
    for h in range(Hb, Hb + Hi):
        for w in range(Wb, Wb + Wi):
            out[h - Hb, w - Wb] = np.sum(kernel * padded_img[h - Hb: Hb + h + 1, w - Wb: Wb + w + 1])
    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    padded_img = zero_pad(image, Hk // 2, Wk // 2)
    strided_padded = as_strided(padded_img, (Hi, Wi, Hk, Wk), strides=2 * padded_img.strides)
    out = np.sum(kernel * strided_padded, axis=(2, 3))
    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_faster(f, np.flip(g))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_faster(f, np.flip(g) - g.mean())


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    padded_img = zero_pad(f, Hk // 2, Wk // 2)
    strided_shape = (Hi, Wi, Hk, Wk)
    strided_view = as_strided(padded_img, strided_shape, strides=2 * padded_img.strides)
    strided_view_mean = strided_view.mean(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    strided_view_std = strided_view.std(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    normalized_strided  = (strided_view - strided_view_mean) / strided_view_std
    template_normalized  = (g - g.mean()) - g.std()
    out = np.sum(normalized_strided  * template_normalized , axis=(2, 3))
    return out