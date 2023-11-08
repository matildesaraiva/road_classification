import os

folder_path = 'C:/Users/LENOVO/Desktop/thesis/raster_pieces/'  # Replace with the actual path to your folder

unique_values = {}

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tif"):
        parts = filename.split("_")
        if len(parts) >= 1:
            prefix = parts[0]
            if prefix not in unique_values:
                unique_values[prefix] = 1
            else:
                unique_values[prefix] += 1

# Print all unique values and their counts
for value, count in unique_values.items():
    print(f"Value: {value}, Count: {count}")