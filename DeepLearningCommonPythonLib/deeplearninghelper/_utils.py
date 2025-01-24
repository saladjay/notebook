
__all__ = ['check_roi_availability', 'get_empyt_structure']

def check_roi_availability(image, roi, debug=False):
    """
    Check if a region of interest (ROI) is within the image bounds
    
    Args:
        image: Image to check
        roi: Region of interest (x_min, y_min, x_max, y_max)
        debug: If True, print debug information
    
    Returns:
        True if ROI is within image bounds, False otherwise
    """
    x_min, y_min, x_max, y_max = roi
    all_values_are_integers = all(isinstance(x, int) for x in [x_min, y_min, x_max, y_max])
    all_values_greater_than_zero = all(x >= 0 for x in [x_min, y_min, x_max, y_max])
    all_edges_are_within_image_bounds = all(x <= l for x, l in zip([x_min, y_min, x_max, y_max], [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
    all_edges_greater_than_zero = all(x > 0 for x in [abs(x_max - x_min), abs(y_max - y_min)])
    if debug:
        print("debuge:", debug)
        print("x_min, y_min, x_max, y_max:", x_min, y_min, x_max, y_max)
        print("all_values_are_integers:", all_values_are_integers)
        print("all_values_greater_than_zero:", all_values_greater_than_zero)
        print("all_edges_are_within_image_bounds:", all_edges_are_within_image_bounds)
        print("all_edges_greater_than_zero:", all_edges_greater_than_zero)
    return all_values_are_integers and all_values_greater_than_zero and all_edges_are_within_image_bounds and all_edges_greater_than_zero


def get_empyt_structure():
    """
    Get an empty structure for storing results
    
    Returns:
        An empty structure for storing results
    """
    return {
  "original_image_path": "",
  "original_image": "",
  "mask_image_path": "",
  "regions": [
    {
      "label": "",
      "crop_filename": "",
      "crop_path": "",
      "bbox": [],
      "mask_path": "",
      "mask_filename": "",
    }
  ]
}