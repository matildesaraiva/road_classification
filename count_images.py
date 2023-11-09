import os

print('Status of the input data that has been created:')
og_vec = 'C:/Users/LENOVO/Desktop/thesis/vector/'
og_ras = 'C:/Users/LENOVO/Desktop/thesis/raster/'
nr_vec = 'C:/Users/LENOVO/Desktop/thesis/vector_pieces/no_road/'
rc_vec = 'C:/Users/LENOVO/Desktop/thesis/vector_pieces/road_center/'
ro_vec = 'C:/Users/LENOVO/Desktop/thesis/vector_pieces/road_other/'
nr_ras = 'C:/Users/LENOVO/Desktop/thesis/raster_pieces/no_road/'
rc_ras = 'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_center/'
ro_ras = 'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_other/'

og_vec_count = len(os.listdir(og_vec))
og_ras_count = len(os.listdir(og_ras))
nr_vec_count = len(os.listdir(nr_vec))
rc_vec_count = len(os.listdir(rc_vec))
ro_vec_count = len(os.listdir(ro_vec))
nr_ras_count = len(os.listdir(nr_ras))
rc_ras_count = len(os.listdir(rc_ras))
ro_ras_count = len(os.listdir(ro_ras))

print('ORIGINALS:')
print(f'Number of original vector files is: {og_vec_count}')
print(f'Number of original raster files is: {og_ras_count}')
print('NO ROAD:')
print(f'Number of vector files is: {nr_vec_count}')
print(f'Number of raster files is: {nr_ras_count}')
print('ROAD CENTER:')
print(f'Number of vector files is: {rc_vec_count}')
print(f'Number of raster files is: {rc_ras_count}')
print('ROAD OTHER:')
print(f'Number of vector files is: {ro_vec_count}')
print(f'Number of raster files is: {ro_ras_count}')

images = len(os.listdir(og_vec))
width = 186
height = 95

nr = nr_vec_count
rc = rc_vec_count
ro = ro_vec_count

print(f'Number of pieces that is expected: {int(images*width*height)}')
print(f'Number of pieces that exists: {int(nr+rc+ro)}')