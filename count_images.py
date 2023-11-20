import os

input = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/mask_groundtruth/'

output_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/no_road/'
output_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/road/'
output_excess_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/no_road/'
output_excess_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/road/'

input_count = len(os.listdir(input))
balanced_no_road_count = len(os.listdir(output_balanced_no_road))
balanced_road_count = len(os.listdir(output_balanced_road))
excess_no_road_count = len(os.listdir(output_excess_no_road))
excess_road_count = len(os.listdir(output_excess_road))

print(f'Number of files in balanced no road: {balanced_no_road_count}')
print(f'Number of files in balanced road: {balanced_road_count}')
print(f'Number of files in excess no road: {excess_no_road_count}')
print(f'Number of files in excess road: {excess_road_count}')

images = len(os.listdir(input))
width = 186
height = 95
total_expected = images*width*height

print(f'Number of pieces expected: {int(total_expected)}')

total_real = balanced_no_road_count + balanced_road_count + excess_no_road_count + excess_road_count
print(f'Number of pieces existing: {int(total_real)}')