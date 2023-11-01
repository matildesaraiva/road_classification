# CRIAÇÃO DE UMA IMAGEM QUE É A COLAGEM DO RASTER COM O MAPA
import numpy as np
import cv2

background = cv2.imread('C:/Users/LENOVO/Desktop/thesis/mask_test/9_3_3.tif')
overlay = cv2.imread('C:/Users/LENOVO/Desktop/thesis/lilp/9_3_3.tif')


# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)
# Save the combined image
cv2.imwrite('C:/Users/LENOVO/Desktop/thesis/901.tif', combined_image)