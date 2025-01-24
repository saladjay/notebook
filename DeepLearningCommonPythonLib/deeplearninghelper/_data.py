import os
import base64
import json
import numpy as np
from pathlib import Path
__all__ = ['array_to_base64', 'base64_to_array', 'image_metadata','region', 'dataset_image']

@staticmethod
def array_to_base64(arr):
    """Convert numpy array to base64 string"""
    try:
        # Convert array to bytes
        arr_bytes = arr.tobytes()
        # Encode bytes as base64 string
        base64_str = base64.b64encode(arr_bytes).decode('utf-8')
        # Include array shape and dtype as metadata
        metadata = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'data': base64_str
        }
        return json.dumps(metadata)
    except Exception as e:
        print(f"Error converting array to base64: {e}")
        return None

@staticmethod
def base64_to_array(base64_str):
    """Convert base64 string back to numpy array"""
    try:
        # Parse metadata
        metadata = json.loads(base64_str)
        # Decode base64 string to bytes
        arr_bytes = base64.b64decode(metadata['data'])
        # Reconstruct array from bytes
        arr = np.frombuffer(arr_bytes, dtype=np.dtype(metadata['dtype']))
        # Reshape array
        arr = arr.reshape(metadata['shape'])
        return arr
    except Exception as e:
        print(f"Error converting base64 to array: {e}")
        return None

class image_metadata:
    """Class for storing image metadata"""

    def __init__(self, filepath=None):
        self.filename = None 
        self.filepath = None
        self.filedata = None

        if filepath is not None:
            if isinstance(filepath, str):
                if os.path.exists(filepath):
                    self.filepath = str(Path(filepath).absolute())
                    self.filename = os.path.basename(self.filepath)
                    self.filedata = None


    def to_dict(self):
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "filedata": array_to_base64(self.filedata) if self.filedata is not None else None
        }
    
    # @classmethod
    def from_dict(self, data_dict):
        self.filename = str(data_dict["filename"])
        self.filepath = str(data_dict["filepath"])
        self.filedata = base64_to_array(data_dict["filedata"]) if data_dict["filedata"] is not None else None


    def update_image_path(self, image_name_to_path_map : dict):
        self.filepath = image_name_to_path_map.get(self.filename, None)

class region:
    def __init__(self):
        self.label = None
        self.bbox = None
        self.partial_image = image_metadata()
        self.mask_image = image_metadata()

    def to_dict(self):
        return {
            "label": self.label,
            "bbox": self.bbox,
            "partial_image": self.partial_image.to_dict(),
            "mask_image": self.mask_image.to_dict(),
        }
    
    def from_dict(self, data_dict):
        self.label = data_dict["label"]
        self.bbox = data_dict["bbox"]
        self.partial_image = image_metadata()
        self.partial_image.from_dict(data_dict["partial_image"])
        self.mask_image = image_metadata()
        self.mask_image.from_dict(data_dict["mask_image"])
    
    def update_image_path(self, image_name_to_path_map : dict):
        self.partial_image.update_image_path(image_name_to_path_map)
        self.mask_image.update_image_path(image_name_to_path_map)

class dataset_image:
    def __init__(self):
        self.image = image_metadata()
        self.mask = image_metadata()
        self.regions = []
        self.json_file_path = None

    def to_dict(self):
        return {
            "image": self.image.to_dict(),
            "mask": self.mask.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
            "json_file_path": str(self.json_file_path)
        }
    
    def from_dict(self, data_dict):
        self.image = image_metadata()
        self.image.from_dict(data_dict["image"])
        self.mask = image_metadata()
        self.mask.from_dict(data_dict["mask"])
        self.regions = [region() for _ in range(len(data_dict["regions"]))]
        for i, region_dict in enumerate(data_dict["regions"]):
            self.regions[i].from_dict(region_dict)
        self.json_file_path = data_dict["json_file_path"]

    def add_region(self, label, bbox : list, partial_image_path : str, mask_image_path : str):
        region = region()
        region.label = label
        region.bbox = bbox
        region.partial_image.filepath = str(Path(partial_image_path).absolute()) if os.path.exists(partial_image_path) else None
        region.partial_image.filename = os.path.basename(region.partial_image.filepath) if region.partial_image.filepath is not None else None

        region.mask_image.filepath = str(Path(mask_image_path).absolute()) if os.path.exists(mask_image_path) else None
        region.mask_image.filename = os.path.basename(region.mask_image.filepath) if region.mask_image.filepath is not None else None

    def update_image_path(self, image_name_to_path_map : dict):
        self.image.update_image_path(image_name_to_path_map)
        self.mask.update_image_path(image_name_to_path_map)
        for region in self.regions:
            region.update_image_path(image_name_to_path_map)


if __name__ == '__main__':
    one_image_metadata = image_metadata(r'D:\works\projects\DefectGenerate\big_shimo\crops\images\1_4_11_17_43_404570_2_region_0.jpg')
    print(one_image_metadata.to_dict())
    d = one_image_metadata.to_dict()
    another_image_metadata = image_metadata()
    print(d)
    another_image_metadata.from_dict(d)
    print(another_image_metadata.to_dict())