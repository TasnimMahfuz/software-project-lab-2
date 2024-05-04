import numpy as np
import cv2
import fhog  # Ensure this module is available and correct

# ffttools
def fftd(img, backwards=False):
    # Apply FFT
    return cv2.dft(
        np.float32(img),
        flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT,
    )

def real(img):
    return img[:, :, 0]

def imag(img):
    return img[:, :, 1]

def complexMultiplication(a, b):
    # Ensure same shape
    res = np.zeros(a.shape, a.dtype)
    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res

def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1.0 / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)
    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res

def rearrange(img):
    # Rearrange image quadrants
    assert img.ndim == 2
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[:yh, :xh], img_[yh:, xh:] = img[yh:, xh:], img[:yh, :xh]
    img_[:yh, xh:], img_[yh:, :xh] = img[yh:, :xh], img[:yh, xh:]
    return img_

# recttools
def x2(rect):
    return rect[0] + rect[2]

def y2(rect):
    return rect[1] + rect[3]

def limit(rect, limit):
    # Limit the rectangle to within boundaries
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect

def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert np.all(np.array(res) >= 0)
    return res

def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = list(window)  # Ensure subscriptable
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])
    assert cutWindow[2] > 0 and cutWindow[3] > 0
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]: cutWindow[1] + cutWindow[3], cutWindow[0]: cutWindow[0] + cutWindow[2]]

    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(
            res, border[1], border[3], border[0], border[2], borderType
        )
    return res

# KCF Tracker Class
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        # Initialization settings
        self.lambdar = 0.0001  # Regularization parameter
        self.padding = 2.5  # Padding around the target
        self.output_sigma_factor = 0.125  # Gaussian output bandwidth

        if hog:
            self.interp_factor = 0.012  # Linear interpolation factor
            self.sigma = 0.6  # Gaussian kernel bandwidth
            self.cell_size = 4  # HOG cell size
            self._hogfeatures = True
        else:
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if multiscale:
            self.template_size = 96
            self.scale_step = 1.05  # Scale step for multiscale
            self.scale_weight = 0.96  # Stability during detection
        elif fixed_window:
            self.template_size = 96
        else:
            self.template_size = 1
        self._tmpl_sz = [0, 0]  # Template size
        self._roi = [0.0, 0.0, 0.0, 0.0]  # Region of interest
        self.size_patch = [0, 0, 0]
        self._scale = 1.0
        self._alphaf = None
        self._prob = None
        self._tmpl = None
        self.hann = None

    def createHanningMats(self):
        # Create Hanning matrices for FFT
        hann2t, hann1t = np.ogrid[0: self.size_patch[0], 0: self.size_patch[1]]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hogfeatures:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d.astype(np.float32)

    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0: sizey, 0: sizex]
        y = (y - syh) ** 2
        x = (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def gaussianCorrelation(self, x1, x2):
        if self._hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(ff
