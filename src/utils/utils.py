import yaml
import numpy as np
import scipy.sparse as sp

from src.utils.logger import TaskLogger


def read_yaml(yaml_directory: str):
    with open(yaml_directory, 'r') as yaml_file:
        return yaml.load(yaml_file, yaml.Loader)
    
def constants(category: str = None, constant: str = None):
    constants_directory = r"src/config/constants.yaml"
    constants = read_yaml(yaml_directory=constants_directory)
    if category is not None and constant is not None:
        return constants[category][constant]
    elif category is not None and constant is None:
        return constants[category]
    elif category is None and constant is None:
        return constants
        
def parameters(category: str = None, parameter: str = None):
    parameters_directory = r"src/config/parameters.yaml"
    parameters = read_yaml(yaml_directory=parameters_directory)
    if category is not None and parameter is not None:
        return parameters[category][parameter]
    elif category is not None and parameter is None:
        return parameters[category]
    elif category is None and parameter is None:
        return parameters
    
def consoleconfigs(config: str = None):
    consoleconfigs_directory = r"src/config/consoleconfigs.yaml"
    consoleconfigs = read_yaml(yaml_directory=consoleconfigs_directory)
    if config is not None:
        return consoleconfigs[config]
    

VIEW_VECTORS = {
    "isometric_top": np.array([1, 1, 1]),
    "isometric_bottom": np.array([1, 1, -1]),
    "dimetric_top": np.array([2, 2, 1]),
    "dimetric_bottom": np.array([2, 2, -1]),
    "right": np.array([0, -1, 0]),
    "left": np.array([0, 1, 0]),
    "front": np.array([1, 0, 0]),
    "back": np.array([-1, 0, 0]),
    "top": np.array([0, 0, 1]),
    "bottom": np.array([0, 0, -1]),
}

logger = TaskLogger("Utils")

def nodeidx_to_dofsidx(nodeidx: np.ndarray) -> np.ndarray:
    """
    Maps node indices to degrees of freedom indices.

    Parameters
    ----------
    nodeidx : (M,) np.ndarray
        Node indices of volumetric mesh with a shape of (M nodes,).

    Returns
    -------
    dofsidx : (3 * M,) np.ndarray
        Degrees of freedom indices with a shape of (3 * M DoFs,).
    """
    dofsidx_idx = np.repeat(nodeidx, 3) * 3
    dofsidx_axis = np.tile(np.array([0, 1, 2]), len(nodeidx))
    return dofsidx_idx + dofsidx_axis

@logger.logprocess("Matrix Slicing")
def sliced_matrix(A: sp.sparray, i: np.ndarray):
    return A[i, :][:, i]

def sliced_vector(b: np.ndarray, i: np.ndarray):
    if b.ndim == 1: return b[i]
    if b.ndim == 2: return b[:, i]
        
if __name__ == "__main__":
    pass