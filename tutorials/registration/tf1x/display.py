# (Provided)
# *** Available as part of UCL MPHY0025 (Information Processing in Medical Imaging) Assessed Coursework 2018-19 ***
# *** This code is with an Apache 2.0 license, University College London ***

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')


def dispImage(img, int_lims = [], ax = None):
  """
  function to display a grey-scale image that is stored in 'standard
  orientation' with y-axis on the 2nd dimension and 0 at the bottom

  INPUTS:   img: image to be displayed
            int_lims: the intensity limits to use when displaying the
               image, int_lims[0] = min intensity to display, int_lims[1]
               = max intensity to display [default min and max intensity
               of image]
            ax: if displaying an image on a subplot grid or on top of a
              second image, optionally supply the axis on which to display 
              the image.
  OUTPUTS:  ax: the axis object after plotting if an axis object is 
            supplied
  """

  #check if intensity limits have been provided, and if not set to min and
  #max of image
  if not int_lims:
    int_lims = [np.nanmin(img), np.nanmax(img)]
    #check if min and max are same (i.e. all values in img are equal)
    if int_lims[0] == int_lims[1]:
      int_lims[0] -= 1
      int_lims[1] += 1
  # take transpose of image to switch x and y dimensions and display with
  # first pixel having coordinates 0,0
  img = img.T
  if not ax:
    plt.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], \
      origin='lower')
  else:
    ax.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], \
      origin='lower')
  #set axis to be scaled equally (assumes isotropic pixel dimensions), tight
  #around the image
  plt.axis('image')
  plt.tight_layout()
  return ax


def dispBinaryImage(binImg, cmap='Greens_r', ax=None):
  """
  function to display a binary image that is stored in 'standard
  orientation' with y-axis on the 2nd dimension and 0 at the bottom

  INPUTS:   binImg: binary image to be displayed
            ax: if displaying an image on a subplot grid or on top of a
              second image, optionally supply the axis on which to display 
              the image.
              E.g. 
              fig = plt.figure()
              ax = fig.gca()
              ax = dispImage(ct_image, ax)
              ax = dispBinaryImage(label_image, ax)
            cmap: color map of the binary image to be displayed 
              (see: https://matplotlib.org/examples/color/colormaps_reference.html)
  OUTPUTS:  ax: the axis object after plotting if an axis object is 
            supplied

  """
  # take transpose of image to switch x and y dimensions and display with
  # first pixel having coordinates 0,0
  binImg = binImg.T
  # set the background pixels to NaNs so that imshow will display
  # transparent 
  binImg = np.where(binImg == 0, np.nan, binImg)
  if not ax:
    plt.imshow(binImg, cmap = cmap, origin='lower')
  else:
    ax.imshow(binImg, cmap = cmap, origin='lower')
  #set axis to be scaled equally (assumes isotropic pixel dimensions), tight
  #around the image
  plt.axis('image')
  plt.tight_layout()
  return ax


def dispImageAndBinaryOverlays(img, bin_imgs = [], bin_cols = [], int_lims = [], ax = None):
  """
  function to display a grey-scale image with one or more binary images
  overlaid

  INPUTS:   img: image to be displayed
            bin_imgs: a list or np.array containing one or more binary images.
                      must have same dimensions as img
            bin_cols: a list or np.array containing the matplotlib colormaps 
                      to use for each binary image E.g. 'Greens_r', 'Reds_r'
                      Must have one colormap for each binary image
            int_lims: the intensity limits to use when displaying the
                      image, int_lims[0] = min intensity to display, int_lims[1]
                      = max intensity to display [default min and max intensity
                      of image]
            ax:       if displaying an image on a subplot grid or on top of a
                      second image, optionally supply the axis on which to display 
                      the image.
  OUTPUTS:  ax:       the axis object after plotting if an axis object is 
                      supplied
  """

  #check if intensity limits have been provided, and if not set to min and
  #max of image
  if not int_lims:
    int_lims = [np.nanmin(img), np.nanmax(img)]
    #check if min and max are same (i.e. all values in img are equal)
    if int_lims[0] == int_lims[1]:
      int_lims[0] -= 1
      int_lims[1] += 1
  # take transpose of image to switch x and y dimensions and display with
  # first pixel having coordinates 0,0
  img = img.T
  if not ax:
    fig = plt.figure()
    ax = fig.gca()
  ax.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], \
    origin='lower')
  for idx, binImg in enumerate(bin_imgs):
    binImg = binImg.T
    # check the binary images and img are the same shape
    if binImg.shape != img.shape:
      print('Error: binary image {} does not have same dimensions as image'.format(idx))
      break
    # set the colormap from bin_cols unless not enough colors have been provided
    try:
      cmap = bin_cols[idx]
    except IndexError:
      cmap = 'Greens_r'
      print('WARNING: not enough colormaps provided - defaulting to Green')
      
    ax.imshow(np.where(binImg == 0, np.nan, binImg), cmap=cmap,\
      origin = 'lower')
  #set axis to be scaled equally (assumes isotropic pixel dimensions), tight
  #around the image
  plt.axis('image')
  plt.tight_layout()
  return ax
