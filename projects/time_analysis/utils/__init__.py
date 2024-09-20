import json
import numpy as np


def load_zones_config(file_path: str) -> list[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    Reads a JSON file containing polygon coordinates and converts them
    into a list of NumPy arrays. Each polygon is represented as a NumPy
    array of coordinates.

    Parameters:
        file_path (str): The path to the JSON configuration file.

    Returns:
        list[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data]
