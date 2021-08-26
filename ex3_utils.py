import numpy as np
import cv2 as cv
from typing import List
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

"""
Given two images, returns the Translation from im1 to im2
:param im1: Image 1
:param im2: Image 2
:param step_size: The image sample size:
:param win_size: The optical flow window size (odd number)
:return: Original points [[x,y]...], [[dU,dV]...] for each points
"""

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):

    kernel = np.array([[-1, 0, 1]])  # (1, 3)
    x_derivative = cv.filter2D(im2, -1, kernel)  # derivative by x
    y_derivative = cv.filter2D(im2, -1, kernel.T)  # derivative by y
    t_subtruct = im2 - im1
    points = []  # points of the image
    u_v = []  # x, y directions
    for row in range(win_size // 2, im1.shape[0] - win_size // 2 + 1, step_size):  # move on the row
        for col in range(win_size // 2, im1.shape[1] - win_size // 2 + 1, step_size):  # move on the column
            Ix = x_derivative[row - win_size // 2: row + win_size // 2, col - win_size // 2: col + win_size // 2]
            Iy = y_derivative[row - win_size // 2: row + win_size // 2, col - win_size // 2: col + win_size // 2]
            It = t_subtruct[row - win_size // 2: row + win_size // 2, col - win_size // 2: col + win_size // 2]
            # concatenate Ix and Iy
            A = np.column_stack((Ix.flatten(), Iy.flatten()))
            AtA = A.T.dot(A)  # matrix size (2,2)
            # need to check the eigenvalues
            lamda_list = np.linalg.eigvals(AtA)
            lamda1 = np.min(lamda_list)
            lamda2 = np.max(lamda_list)
            if lamda1 > 1 and lamda2 / lamda1 < 100:
                arr = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
                u_v.append(np.linalg.inv(AtA).dot(arr))
                points.append([col, row])
    return np.array(points), np.array(u_v)


"""
claculate the gaussian kernel 
"""


def get_gaussian_kernel():
    kernel = cv.getGaussianKernel(5, -1)
    kernel = kernel.dot(kernel.T)
    return kernel


"""
Creates a Gaussian Pyramid
:param img: Original image
:param levels: Pyramid depth
:return: Gaussian pyramid (list of images)
"""


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    res_p = []
    width = (2 ** levels) * int(img.shape[1] / (2 ** levels))
    height = (2 ** levels) * int(img.shape[0] / (2 ** levels))
    img = cv.resize(img, (width, height))  # resize the image to dimensions that can be divided into 2 x times
    img = img.astype(np.float64)
    res_p.append(img)
    for dep in range(levels):
        filter_img = cv.filter2D(res_p[dep], -1, get_gaussian_kernel()/(np.sum(get_gaussian_kernel())))
        reduce_img = filter_img[::2, ::2]  # reduce image size to half of the size
        res_p.append(reduce_img)
    return res_p


"""
Expands a Gaussian pyramid level one step up
:param img: Pyramid image at a certain level
:param gs_k: The kernel to use in expanding
:return: The expanded level
"""


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if img.ndim == 3:  # RGB img
        padded_im = np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3))
    else:
        padded_im = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    padded_im[::2, ::2] = img
    return cv.filter2D(padded_im, -1, gs_k)


"""
Creates a Laplacian pyramid
:param img: Original image
:param levels: Pyramid depth
:return: Laplacian Pyramid (list of images)
"""


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gauss_pyr = gaussianPyr(img, levels)
    g_kernel = get_gaussian_kernel() / (
                np.sum(get_gaussian_kernel()) * 0.25)  # the sum of the down sampling need to be 4
    for i in range(levels - 1):
        gauss_pyr[i] = gauss_pyr[i] - gaussExpand(gauss_pyr[i + 1], g_kernel)
    return gauss_pyr


"""
Resotrs the original image from a laplacian pyramid
:param lap_pyr: Laplacian Pyramid
:return: Original image
"""


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    levels = len(lap_pyr)
    temp = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gs_k = get_gaussian_kernel() / (np.sum(get_gaussian_kernel())*0.25)
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(temp, gs_k)
        temp = expand + lap_pyr[i - 1]
        i -= 1
    return temp


"""
Blends two images using PyramidBlend method
:param img_1: Image 1
:param img_2: Image 2
:param mask: Blend mask
:param levels: Pyramid depth
:return: Blended Image
"""


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    img_1, img_2 = resize_as_mask(img_1, img_2, mask)  # check that the resize of the mask equal to the images
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    l_a = laplaceianReduce(img_1, levels)
    l_b = laplaceianReduce(img_2, levels)
    g_m = gaussianPyr(mask, levels)
    ans = (l_a[levels - 1] * g_m[levels - 1] + (1 - g_m[levels - 1]) * l_b[levels - 1])
    gs_k = get_gaussian_kernel() / np.sum(get_gaussian_kernel() * 0.25)
    k = levels - 2
    while k >= 0:  # go through the list from end to start
        ans = gaussExpand(ans, gs_k) + l_a[k] * g_m[k] + (1 - g_m[k]) * l_b[k]
        k -= 1
    naive_blend = cv.resize(naive_blend, (ans.shape[1], ans.shape[0]))
    return naive_blend, ans


"""
change the size of the images as the mask 
"""


def resize_as_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    new_width = mask.shape[1]
    new_height = mask.shape[0]
    img1 = cv.resize(img1, (new_width, new_height))
    img2 = cv.resize(img2, (new_width, new_height))
    return img1, img2
