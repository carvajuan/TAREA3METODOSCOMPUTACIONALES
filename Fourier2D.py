import numpy as np
import matplotlib.pylab as plt
from scipy import fftpack
from scipy import ndimage
from PIL import Image
from matplotlib.colors import LogNorm

im = plt.imread('arbol.png').astype(float) #SE LEE Y ALMACENA LA IMAGEN COMO UN ARRAY DE FLOATS
