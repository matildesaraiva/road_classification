##########################################################################################
##########################################################################################
# CRIAÇÃO DE UMA IMAGEM QUE É A COLAGEM DO RASTER COM O MAPA
import numpy as np
import cv2

# This part is going to be deleted, it only has the purpose of making it easier to access if both images match
background = cv2.imread('piece_0_6.tif')
overlay = cv2.imread('C:/data/vector.tif')
# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)
# Save the combined image
cv2.imwrite('C:/data/combined.tif', combined_image)


##########################################################################################
##########################################################################################
# CONFIRMAR SE O ARRAY ESTÁ EM BINÁRIO E FOI CORRETAMENTE EXTRAÍDO
import rasterio
import numpy as np
import cv2
vec = rasterio.open("PATH/TO/FILE")
image = vec.read()
print(image.shape)
image[0][1]

##########################################################################################
##########################################################################################
# ENGROSSAR A LINHA DO VECTOR (DUPLICAR E GUARDAR UMA NOVA IMAGEM)

modified_image = np.copy(image)
for row in range(image.shape[1]):
    for col in range(1, image.shape[2] - 1):
        pixel_value = image[0, row, col]
        if pixel_value == 1:
            modified_image[0, row, col - 1] = 1
            modified_image[0, row, col + 1] = 1
modified_image[0][1]
output_path = 'C:/data/vector.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=modified_image.shape[1],
    width=modified_image.shape[2],
    count=1,
    dtype=modified_image.dtype,
    crs=rasterio.crs.CRS.from_epsg(4326),
    transform=vec.meta['transform']) as dst:
    # Write the modified_image array to the raster dataset
    dst.write(modified_image)