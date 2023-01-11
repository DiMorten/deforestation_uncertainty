import matplotlib.pyplot as plt
import numpy as np
import cv2
from icecream import ic



# ========================
# crop samples

def plotCropSample(im1, im2, im3, lims = None, 
        titles = ['Optical', 'Secondary Veg.', 'Past deforestation'],
        cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray]):
    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    if lims is None:
        axes[0].imshow(im1, cmap=cmaps[0])
        axes[1].imshow(im2, cmap=cmaps[1])
        axes[2].imshow(im3, cmap=cmaps[2])
    else:
        axes[0].imshow(im1[lims[0]:lims[1], lims[2]:lims[3]], cmap=cmaps[0])
        axes[1].imshow(im2[lims[0]:lims[1], lims[2]:lims[3]], cmap=cmaps[1])
        axes[2].imshow(im3[lims[0]:lims[1], lims[2]:lims[3]], cmap=cmaps[2])

    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')

    axes[0].title.set_text(titles[0])
    axes[1].title.set_text(titles[1])
    axes[2].title.set_text(titles[2])

def applyBackgroundMask(im, label):
    im = im.copy()
    im[label == 2] = 0
    return im

def invertMaskFromIm(im):
    im = np.abs(1 - im)
    return im
from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_im(im, ax, title = "", cmap = "jet"):
    im_plt = ax.imshow(im.astype(np.float32), cmap = cmap)
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_plt, cax=cax) 

def plotCropSample4(im1, im2, im3, im4, lims = None, 
        titles = ['Optical', '.', 'Secondary Veg.', 'Past deforestation'],
        cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray, plt.cm.gray],
        maskBackground = [False, False, False, False],
        invertMask = [False, False, False, False],
        uncertainty_vlims = [0, 1], colorbar = False):
    fig, axes = plt.subplots(1, 4)
    fig.set_figheight(14)
    fig.set_figwidth(14)
    
    if lims is not None:
        im1 = im1[lims[0]:lims[1], lims[2]:lims[3]]
        im2 = im2[lims[0]:lims[1], lims[2]:lims[3]]
        im3 = im3[lims[0]:lims[1], lims[2]:lims[3]]
        im4 = im4[lims[0]:lims[1], lims[2]:lims[3]]



    if maskBackground[1] == True:
        im2 = applyBackgroundMask(im2, im4)
    if maskBackground[2] == True:
        im3 = applyBackgroundMask(im3, im3)
    if maskBackground[3] == True:
        im4 = applyBackgroundMask(im4, im4)

    if invertMask[1] == True:
        im2 = invertMaskFromIm(im2)
    if invertMask[2] == True:
        im3 = invertMaskFromIm(im3)
    if invertMask[3] == True:
        im4 = invertMaskFromIm(im4)


    axes[0].imshow(im1, cmap=cmaps[0])
    plot = axes[1].imshow(im2, cmap=cmaps[1], vmin=0, vmax=1)
    if colorbar == True:
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax, orientation="horizontal")


    axes[2].imshow(im3, cmap=cmaps[2])
    plot = axes[3].imshow(im4, cmap=cmaps[3], vmin=uncertainty_vlims[0], vmax=uncertainty_vlims[1])
    if colorbar == True:
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax, orientation="horizontal") 


    axes[0].axis('off')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    axes[0].title.set_text(titles[0])
    axes[1].title.set_text(titles[1])
    axes[2].title.set_text(titles[2])
    axes[3].title.set_text(titles[3])  


def plotCropSample5(im1, im2, im3, im4, im5, lims = None, 
        titles = ['Optical', '.', '.', 'Secondary Veg.', 'Past deforestation'],
        cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray, plt.cm.gray, plt.cm.gray],
        maskBackground = [False, False, False, False, False],
        invertMask = [False, False, False, False, False],
        uncertainty_vlims = [0, 1], colorbar = False):
    fig, axes = plt.subplots(1, 5)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    
    if lims is not None:
        im1 = im1[lims[0]:lims[1], lims[2]:lims[3]]
        im2 = im2[lims[0]:lims[1], lims[2]:lims[3]]
        im3 = im3[lims[0]:lims[1], lims[2]:lims[3]]
        im4 = im4[lims[0]:lims[1], lims[2]:lims[3]]
        im5 = im5[lims[0]:lims[1], lims[2]:lims[3]]


    if maskBackground[2] == True:
        im3 = applyBackgroundMask(im3, im5)
    if maskBackground[3] == True:
        im4 = applyBackgroundMask(im4, im4)
    if maskBackground[4] == True:
        im5 = applyBackgroundMask(im5, im5)

    if invertMask[2] == True:
        im3 = invertMaskFromIm(im3)
    if invertMask[3] == True:
        im4 = invertMaskFromIm(im4)
    if invertMask[4] == True:
        im5 = invertMaskFromIm(im5)


    axes[0].imshow(im1, cmap=cmaps[0])
    axes[1].imshow(im2, cmap=cmaps[0])
    plot = axes[2].imshow(im3, cmap=cmaps[2], vmin=0, vmax=1)
    if colorbar == True:
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax, orientation="horizontal")


    axes[3].imshow(im4, cmap=cmaps[3])
    plot = axes[4].imshow(im5, cmap=cmaps[4], vmin=uncertainty_vlims[0], vmax=uncertainty_vlims[1])
    if colorbar == True:
        divider = make_axes_locatable(axes[4])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax, orientation="horizontal") 


    for idx, title in enumerate(titles):
        if idx == 0 or idx == 1:
            axes[idx].axis('off')
        else:
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])


    for idx, title in enumerate(titles):
        axes[idx].title.set_text(title)

    plt.savefig('Para' + ' normalized RGB.png', dpi=300, bbox_inches='tight')

def whiteBckndImShow(im, figsize = (10,10)):
    im = np.abs(1 - im)
    '''
    im = im.astype(np.uint8)*255
    
    im = np.repeat(
        np.expand_dims(im, axis=-1),
        3, axis = -1)
    '''
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im, cmap = plt.cm.gray)
    return fig


def getRgbErrorMask(predicted, label):
    false_positive_mask = predicted - label

    error_mask_to_show = predicted.copy()
    error_mask_to_show[false_positive_mask == 1] = 2
    error_mask_to_show[false_positive_mask == -1] = 3
    return error_mask_to_show
def saveRgbErrorMask(error_mask_to_show, dim = None):


    colormap = np.array([[0, 0, 0],
            [255, 255, 255],
            [0, 0, 255],
            [255, 0, 0]])

    colormap = np.array([[255, 255, 255],
            [0, 0, 0],
            [45, 150, 255],
            [255, 146, 36]])


    
    error_mask_to_show_rgb=cv2.cvtColor(error_mask_to_show,cv2.COLOR_GRAY2RGB)
    error_mask_to_show_rgb_tmp = error_mask_to_show_rgb.copy()
    for idx in range(colormap.shape[0]):
        for chan in range(error_mask_to_show_rgb.shape[-1]):
            error_mask_to_show_rgb[...,chan][error_mask_to_show_rgb_tmp[...,chan] == idx]=colormap[idx,chan]

    error_mask_to_show_rgb=cv2.cvtColor(error_mask_to_show_rgb,cv2.COLOR_BGR2RGB)
    if dim is not None:
        error_mask_to_show_rgb = cv2.resize(error_mask_to_show_rgb, 
            dim, interpolation = cv2.INTER_NEAREST)
    return error_mask_to_show_rgb


from mpl_toolkits.axes_grid1 import make_axes_locatable

epsilon = 1e-15
def show_im(im, ax, title = "", cmap = "jet"):
    im_plt = ax.imshow(im.astype(np.float32), cmap = cmap)
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_plt, cax=cax) 
