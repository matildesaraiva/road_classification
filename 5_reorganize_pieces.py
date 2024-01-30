import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image

# Abrir uma imagem satélite
image = cv2.imread('C:/Users/LENOVO/Desktop/thesis/data/1_data/raster/border/0.tif')

# adicionar uma overlay
image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
overlay = np.zeros_like(image_with_alpha, dtype=np.uint8)
overlay[:, :, 3] = 255  # 0 represents fully transparent, 255 represents fully opaque
result = cv2.addWeighted(image_with_alpha, 1, overlay, 1, 0)

# Save the image
cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', result)
# pôr a overlay transparente
# dizer que os primeiros 64x64 são vermelho
# depois ir quadrado a quadrado

