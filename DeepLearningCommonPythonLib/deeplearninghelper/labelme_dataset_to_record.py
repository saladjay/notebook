import os
import glob
import json
import numpy as np
import cv2
from pathlib import Path
from ._utils import check_roi_availability, get_empyt_structure
from ._data import dataset_image, region, image_metadata
def labelme_dataset_to_record(json_dir, image_dir, output_file):
    """
    Convert a LabelMe dataset to json format.

    Args:    
        json_dir (str): Path to the directory containing the JSON files.
        image_dir (str): Path to the directory containing the image files.
        output_file (str): Path to the output json file.
    """
    if not os.path.exists(json_dir):
        raise ValueError("Directory does not exist: {}".format(json_dir))
    if not os.path.exists(image_dir):
        raise ValueError("Directory does not exist: {}".format(image_dir))

    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    json_to_image_map = {}  
    for json_file in json_dir.glob('*.json'):
        json_to_image_map[json_file.stem] = [json_file, None]

    for image_file in os.listdir(str(image_dir)):
        if Path(image_file).stem in json_to_image_map and Path(image_file).suffix in ['.jpg', '.png', '.jpeg']:
            json_to_image_map[Path(image_file).stem][1] = os.path.join(str(image_dir), image_file)


    datasets = []
    for key, value in json_to_image_map.items():
        if value[1] is None:
            print("Warning: No image found for {}".format(key))
        else:
            one_dataset_image = dataset_image()
            one_dataset_image.json_file_path = value[0]
            one_dataset_image.image = image_metadata(value[1])
            one_dataset_image.regions = []  
            image = cv2.imread(value[1])
            if image is None:
                print("Warning: Image {} not found".format(value[1]))
                continue
            with open(value[0], 'r') as f:
                json_data = json.load(f)
                for i, shape in enumerate(json_data['shapes']):
                    label = shape['label']
                    points = np.array(shape['points'], dtype=np.int32)
        
                    # Get bounding box
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)

                    # Check if ROI is within image bounds
                    if not check_roi_availability(image, (x_min, y_min, x_max, y_max), debug=False):
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(image.shape[1], x_max)
                        y_max = min(image.shape[0], y_max)
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        if not check_roi_availability(image, (x_min, y_min, x_max, y_max), debug=False):
                            print(f"Skipping region {i} with invalid bounds: {x_min}, {y_min}, {x_max}, {y_max}, image shape: {image.shape}")
                            check_roi_availability(image, (x_min, y_min, x_max, y_max), debug=True)
                            continue
                    one_region = region()
                    one_region.label = label
                    one_region.bbox = [x_min, y_min, x_max, y_max]
                    one_dataset_image.regions.append(one_region)
            datasets.append(one_dataset_image.to_dict())
    if os.path.exists(output_file):
        try:
            datasets = json.load(open(output_file, 'r')) + datasets
        except:
            pass
    with open(output_file, 'w') as f:
        # print(datasets)
        json.dump(datasets, f, indent=4)

def update_image_path(dataset_json_path, dataset_image_dir):
    """
    Update the image path in the dataset json file.

    Args:    
        dataset_json_path (str): Path to the dataset json file.
        dataset_image_dir (str): Path to the directory containing the image files.
    """
    image_name_to_path_map = {}
    for root, dirs, files in os.walk(dataset_image_dir):

        for file in files:
            if Path(file).suffix in ['.jpg', '.png', '.jpeg']:
                image_name_to_path_map[file] = os.path.join(root, file)


    dataset = json.load(open(dataset_json_path, 'r'))

    for idx in range(len(dataset)):
        image_info = dataset[idx]
        one_dataset_image = dataset_image()
        one_dataset_image.from_dict(image_info)
        one_dataset_image.update_image_path(image_name_to_path_map)
        dataset[idx] = one_dataset_image.to_dict()

    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        

if __name__ == '__main__':
    json_dir = r'D:\works\projects\DefectGenerate\big_shimo\val\jsons'
    image_dir = r'D:\works\projects\DefectGenerate\big_shimo\val\images'
    output_file = r'D:\works\projects\DefectGenerate\big_shimo\dataset.json'
    labelme_dataset_to_record(json_dir, image_dir, output_file)