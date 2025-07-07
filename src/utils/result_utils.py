import os
from typing import Literal
import copy
import itertools

import numpy as np
import pyvista as pv
import pandas as pd
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_array, csr_array, load_npz
from pprint import pprint
from tabulate import tabulate
from tqdm import tqdm

from src.utils.utils import consoleconfigs, VIEW_VECTORS
from src.utils.logger import TaskLogger


class DeformationCurveResultHandler:
    
    logger = TaskLogger("DeformationCurveResultHandler")
    
    def __init__(self) -> None:
        
        self.mean_measured_result_points = None
        self.simulation_result_points = None
        
        self.youngs_modulus = None
        self.poissons_ratio = None
        
        self.mean_relative_error_horz = None
        self.mean_relative_error_vert = None
        self.max_relative_error_horz = None
        self.max_relative_error_vert = None
        
        self.mean_absolute_error = None
        self.max_absolute_error = None
        self.sum_absolute_error = None
        
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.max_squared_error = None
        self.sum_squared_error = -1
        
        self.error_data = {
            "youngs_modulus": [],
            "poissons_ratio": [],
            "mean_relative_error_horz": [],
            "mean_relative_error_vert": [],
            "max_relative_error_horz": [],
            "max_relative_error_vert": [],
            "mean_absolute_error": [],
            "max_absolute_error": [],
            "sum_absolute_error": [],
            "mean_squared_error": [],
            "root_mean_squared_error": [],
            "max_squared_error": [],
            "sum_squared_error": [],
        }
        self.df_error_data = pd.DataFrame(self.error_data)

    def show_data(self):
        pprint(f"\nSimulation Result Points:\n{self.simulation_result_points}")
        pprint(f"\nMeasured Result Points:\n{self.mean_measured_result_points}")
        
    def set_youngs_modulus(self, youngs_modulus):
        self.youngs_modulus = youngs_modulus
        
    def set_poissons_ratio(self, poissons_ratio):
        self.poissons_ratio = poissons_ratio
        
    def set_simulation_result_points(self, simulation_result_points):
        self.simulation_result_points = simulation_result_points
        
    def set_raw_measured_result_points(self, raw_measured_result_points_top: np.ndarray, raw_measured_result_points_bottom: np.ndarray):
        
        # because the measured points from top side is always existed,
        # so just check that the bottom side is empty or not
        self.raw_measured_result_points_top = raw_measured_result_points_top
        if len(raw_measured_result_points_bottom) == 0:
            self.raw_measured_result_points_bottom = np.array([]).reshape((3, -1, 3))
        elif len(raw_measured_result_points_bottom) > 0:
            self.raw_measured_result_points_bottom = raw_measured_result_points_bottom
        
        self.raw_measured_result_points_1 = np.vstack([self.raw_measured_result_points_top[0], self.raw_measured_result_points_bottom[0]])
        self.raw_measured_result_points_2 = np.vstack([self.raw_measured_result_points_top[1], self.raw_measured_result_points_bottom[1]])
        self.raw_measured_result_points_3 = np.vstack([self.raw_measured_result_points_top[2], self.raw_measured_result_points_bottom[2]])

    def set_mean_measured_result_points(self, mean_measured_result_points):
        self.mean_measured_result_points = mean_measured_result_points
        
    def set_non_value_axis(self, axis: Literal["x", "y", "z"]):
        if axis == "x":
            self.valued_axis = np.array([1, 2])
        elif axis == "y":
            self.valued_axis = np.array([0, 2])
        elif axis == "z":
            self.valued_axis = np.array([0, 1])
    
    def calculate_error(self):
        
        # create constant and instances of true and predicted results
        n = min(len(self.mean_measured_result_points), len(self.simulation_result_points))
        res_true = self.mean_measured_result_points[0:n]
        res_pred = self.simulation_result_points[0:n]
        
        # raw error and relative error are represented by euclidean distance between two results
        # raw_error, relative_error_horz, and relative_error_vert are 1D np.ndarray shape of n
        raw_error = np.zeros(n, dtype=float)
        relative_error_horz = np.zeros(n, dtype=float)
        relative_error_vert = np.zeros(n, dtype=float)
        for i in range(0, n):
            # avoid 0 / 0 = NaN during calculate relative error
            if i == 0 or i == 100:
                continue
            else:
                raw_error[i] = np.linalg.norm(res_true[i] - res_pred[i])
                ax0 = self.valued_axis[0]
                ax1 = self.valued_axis[1]
                relative_error_horz[i] = np.abs((res_true[i, ax0] - res_pred[i, ax0]) / (res_true[i, ax0]))
                relative_error_vert[i] = np.abs((res_true[i, ax1] - res_pred[i, ax1]) / (res_true[i, ax1]))
        
        # calculate various type of error from an array of raw error to be one number that describe the error characteristics
        absolute_error = np.abs(raw_error)
        squared_error = np.square(raw_error)
        
        self.mean_relative_error_horz = np.mean(relative_error_horz)
        self.mean_relative_error_vert = np.mean(relative_error_vert)
        self.max_relative_error_horz = np.max(relative_error_horz)
        self.max_relative_error_vert = np.max(relative_error_vert)
        
        self.mean_absolute_error = np.mean(absolute_error)
        self.max_absolute_error = np.max(absolute_error)
        self.sum_absolute_error = np.sum(absolute_error)
        
        self.mean_squared_error = np.mean(squared_error)
        self.root_mean_squared_error = np.sqrt(self.mean_squared_error)
        self.max_squared_error = np.max(squared_error)
        self.sum_squared_error = np.sum(squared_error)
                
    def update_error(self):
        
        # put new calculated error into data dictionary
        # pprint(f"raw error:\n{self.raw_error}")
        self.error_data = {
            "youngs_modulus": self.youngs_modulus,
            "poissons_ratio": self.poissons_ratio,
            "mean_relative_error_horz": self.mean_relative_error_horz,
            "mean_relative_error_vert": self.mean_relative_error_vert,
            "max_relative_error_horz": self.max_relative_error_horz,
            "max_relative_error_vert": self.max_relative_error_vert,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "sum_absolute_error": self.sum_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "max_squared_error": self.max_squared_error,
            "sum_squared_error": self.sum_squared_error,
        }

        # add new data to the new last row
        self.df_error_data.loc[len(self.df_error_data)] = self.error_data
        
        # print the new added data
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            self.logger.loginfo(f"\n{self.df_error_data.iloc[-1]}")
        
    def save_deformation_curve_data(self, path):
        self.deformation_curve_data = {
            "x_simulated": self.simulation_result_points[:, 0],
            "y_simulated": self.simulation_result_points[:, 1],
            "z_simulated": self.simulation_result_points[:, 2],
            "x_measured_raw_1": self.raw_measured_result_points_1[:, 0],
            "y_measured_raw_1": self.raw_measured_result_points_1[:, 1],
            "z_measured_raw_1": self.raw_measured_result_points_1[:, 2],
            "x_measured_raw_2": self.raw_measured_result_points_2[:, 0],
            "y_measured_raw_2": self.raw_measured_result_points_2[:, 1],
            "z_measured_raw_2": self.raw_measured_result_points_2[:, 2],
            "x_measured_raw_3": self.raw_measured_result_points_3[:, 0],
            "y_measured_raw_3": self.raw_measured_result_points_3[:, 1],
            "z_measured_raw_3": self.raw_measured_result_points_3[:, 2],
            "x_measured_mean": self.mean_measured_result_points[:, 0],
            "y_measured_mean": self.mean_measured_result_points[:, 1],
            "z_measured_mean": self.mean_measured_result_points[:, 2],
        }
        self.df_deformation_curve_data = pd.DataFrame(self.deformation_curve_data)
        self.df_deformation_curve_data.to_csv(path)
        self.logger.loginfo(f"Data of deformation curve is successfully saved at {path}")
    
    def save_error_data(self, path):
        self.df_error_data.to_csv(path)
        print(f"{path} is successfully saved.")


class MarkersResultHandler:
    
    logger = TaskLogger("MarkersResultHandler")
    
    def __init__(self) -> None:
        
        self.simulated_markers_initial = None
        self.simulated_markers_deformed = None
        self.measured_markers_initial = None
        self.measured_markers_deformed = None
        
        self.mean_measured_result_points = None
        self.simulation_result_points = None
        
        self.youngs_modulus = None
        self.poissons_ratio = None
        
        self.mean_relative_error_horz = None
        self.mean_relative_error_vert = None
        self.max_relative_error_horz = None
        self.max_relative_error_vert = None
        
        self.mean_absolute_error = None
        self.max_absolute_error = None
        self.sum_absolute_error = None
        
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.max_squared_error = None
        self.sum_squared_error = -1
        
        self.error_data = {
            "youngs_modulus": [],
            "poissons_ratio": [],
            "mean_relative_error_horz": [],
            "mean_relative_error_vert": [],
            "max_relative_error_horz": [],
            "max_relative_error_vert": [],
            "mean_absolute_error": [],
            "max_absolute_error": [],
            "sum_absolute_error": [],
            "mean_squared_error": [],
            "root_mean_squared_error": [],
            "max_squared_error": [],
            "sum_squared_error": [],
        }
        self.df_error_data = pd.DataFrame(self.error_data)

    def show_data(self):
        pprint(f"\nSimulation Result Points:\n{self.simulation_result_points}")
        pprint(f"\nMeasured Result Points:\n{self.mean_measured_result_points}")
        
    def set_youngs_modulus(self, youngs_modulus):
        self.youngs_modulus = youngs_modulus
        
    def set_poissons_ratio(self, poissons_ratio):
        self.poissons_ratio = poissons_ratio

    def set_simulated_markers(self, initial_markers: np.ndarray, deformed_markers: np.ndarray):
        self.simulated_markers_initial = initial_markers
        self.simulated_markers_deformed = deformed_markers
        
    def set_measured_markers(self, initial_markers: np.ndarray, deformed_markers: np.ndarray):
        self.measured_markers_initial = initial_markers
        self.measured_markers_deformed = deformed_markers
        
    def save_error_data(self, path):
        
        simulated_markers_displacement = self.simulated_markers_deformed - self.simulated_markers_initial
        simulated_markers_euclidean_distance = np.sqrt(np.sum((simulated_markers_displacement) ** 2, axis=1))
        simulated_markers_displacement = simulated_markers_displacement.flatten()
        simulated_markers_euclidean_distance = np.repeat(simulated_markers_euclidean_distance, 3)

        measured_markers_displacement = self.measured_markers_deformed - self.measured_markers_initial
        measured_markers_euclidean_distance = np.sqrt(np.sum((measured_markers_displacement) ** 2, axis=1))
        measured_markers_displacement = measured_markers_displacement.flatten()
        measured_markers_euclidean_distance = np.repeat(measured_markers_euclidean_distance, 3)
        
        raw_error = self.measured_markers_deformed.flatten() - self.simulated_markers_deformed.flatten()
        absolute_error = np.abs(raw_error)
        squared_error = np.square(raw_error)
        
        mean_absolute_error = np.mean(absolute_error)
        max_absolute_error = np.max(absolute_error)
        sum_absolute_error = np.sum(absolute_error)
        
        mean_squared_error = np.mean(squared_error)
        root_mean_squared_error = np.sqrt(mean_squared_error)
        max_squared_error = np.max(squared_error)
        sum_squared_error = np.sum(squared_error)
        
        self.error_data = {
            "markers": [
                "fixed_1_x", "fixed_1_y", "fixed_1_z", 
                "fixed_2_x", "fixed_2_y", "fixed_2_z", 
                "float_1_x", "float_1_y", "float_1_z", 
                "float_2_x", "float_2_y", "float_2_z", 
                "float_3_x", "float_3_y", "float_3_z", 
                "float_4_x", "float_4_y", "float_4_z",
            ],
            "simulated_initial": self.simulated_markers_initial.flatten(),
            "simulated_deformed": self.simulated_markers_deformed.flatten(),
            "measured_initial": self.measured_markers_initial.flatten(),
            "measured_deformed": self.measured_markers_deformed.flatten(),

            "simulated_markers_displacement": simulated_markers_displacement,
            "simulated_markers_euclidean_distance": simulated_markers_euclidean_distance,
            "measured_markers_displacement": measured_markers_displacement,
            "measured_markers_euclidean_distance": measured_markers_euclidean_distance,
            
            "raw_error": raw_error,
            "absolute_error": absolute_error,   
            "squared_error": squared_error,     
            "mean_absolute_error": [mean_absolute_error] * len(self.simulated_markers_initial.flatten()),
            "max_absolute_error": [max_absolute_error] * len(self.simulated_markers_initial.flatten()),
            "sum_absolute_error": [sum_absolute_error] * len(self.simulated_markers_initial.flatten()),
            "mean_squared_error": [mean_squared_error] * len(self.simulated_markers_initial.flatten()),
            "root_mean_squared_error": [root_mean_squared_error] * len(self.simulated_markers_initial.flatten()),
            "max_squared_error": [max_squared_error] * len(self.simulated_markers_initial.flatten()),
            "sum_squared_error": [sum_squared_error] * len(self.simulated_markers_initial.flatten()),
        }
        self.df_error_data = pd.DataFrame(self.error_data)
        self.df_error_data.to_csv(path)
        print(f"{path} is successfully saved.")
        

class BoundsResultHandler:

    logger = TaskLogger("BoundsResultHandler")

    def __init__(self) -> None:
        self.initial_bounds = None
        self.deformed_bounds = None

    def set_bounds(self, initial, deformed):
        if initial is not None:
            self.initial_bounds = np.array(initial)
        if deformed is not None:
            self.deformed_bounds = np.array(deformed)

    def calc_displacements(self):

        self.displacements = np.zeros((3, 3), dtype=float)
        self.magnitudes = np.zeros((3,), dtype=float)
        # [negative, positive, total]
        
        bounds_displacement = self.deformed_bounds - self.initial_bounds
        
        self.displacements[0, :] = np.array([bounds_displacement[0], bounds_displacement[2], bounds_displacement[4]])
        self.displacements[1, :] = np.array([bounds_displacement[1], bounds_displacement[3], bounds_displacement[5]])

        self.magnitudes[0] = np.linalg.norm(self.displacements[0, :])
        self.magnitudes[1] = np.linalg.norm(self.displacements[1, :])

        self.displacements[2, :] = np.array([bounds_displacement[1] - bounds_displacement[0], bounds_displacement[3] - bounds_displacement[2], bounds_displacement[5] - bounds_displacement[4]])
        self.magnitudes[2] = np.linalg.norm(self.displacements[2, :])

    def get_displacements(self):
        self.calc_displacements()
        return  self.displacements, self.magnitudes
    

class DomainsResultHandler:

    logger = TaskLogger("DomainsResultHandler")

    def __init__(self):
        self.simulated_initial = None
        self.simulated_deformed = None
        self.simulated_displacement = None
        self.domain_volumes = np.array([None] * 4)
        self.is_measured_domains_set = False
        self.max_magnitudes = None
        self.mean_magnitudes = None
        self.stddev_magnitudes = None
        self.hausdorff_distances = np.array([None] * 3)

    def extract_largest_domain(self, grid):
        # since this handler often uses with domain that is included the embedded domain
        # so when we calculate the hausdorff distance of the domain,
        # we don't want to consider the embedded domain
        # so, we need to acheive the surface mesh that is excluded the embedded domain
        surf = grid.extract_surface().connectivity(extraction_mode="largest")
        surf.set_active_scalars("RegionId")
        largest_surf = surf.threshold(0)
        return largest_surf
    
    def set_simulated_domains(self, simulated_initial: pv.UnstructuredGrid, simulated_deformed: pv.UnstructuredGrid, simulated_displacement: np.ndarray):
        self.simulated_initial = self.extract_largest_domain(simulated_initial)
        self.simulated_deformed = self.extract_largest_domain(simulated_deformed)
        self.simulated_displacement = simulated_displacement        
        
    def set_measured_domains(self, measured_initial: pv.UnstructuredGrid, measured_deformed: pv.UnstructuredGrid):
        self.measured_initial = self.extract_largest_domain(measured_initial)
        self.measured_deformed = self.extract_largest_domain(measured_deformed)
        self.is_measured_domains_set = True
    
    def get_simulated_domain_volumes(self):        
        self.domain_volumes[0] = self.simulated_initial.volume
        self.domain_volumes[1] = self.simulated_deformed.volume
        if self.is_measured_domains_set:
            self.domain_volumes[2] = self.measured_initial.volume
            self.domain_volumes[3] = self.measured_deformed.volume
        return self.domain_volumes

    def get_nodal_displacements(self):
        magnitudes = np.linalg.norm(self.simulated_displacement, axis=1)
        self.max_magnitudes = np.amax(magnitudes)
        self.mean_magnitudes = np.mean(magnitudes)
        self.stddev_magnitudes = np.std(magnitudes)
        return self.max_magnitudes, self.mean_magnitudes, self.stddev_magnitudes

    def get_hausdorff_distances(self):
        simulated_initial_surf = self.simulated_initial.extract_surface()
        simulated_deformed_surf = self.simulated_deformed.extract_surface()
        self.hausdorff_distances[0] = directed_hausdorff(simulated_initial_surf.points, simulated_deformed_surf.points)[0]
        
        if self.is_measured_domains_set:
            measured_initial_surf = self.measured_initial.extract_surface()
            measured_deformed_surf = self.measured_deformed.extract_surface()
            self.hausdorff_distances[1] = directed_hausdorff(measured_initial_surf.points, measured_deformed_surf.points)[0]
            self.hausdorff_distances[2] = directed_hausdorff(simulated_deformed_surf.points, measured_deformed_surf.points)[0]
        
        return self.hausdorff_distances
        
  
class LesionResultHandler:

    logger = TaskLogger("LesionResultHandler")

    def __init__(self) -> None:
        self.simulated_initial_node = None
        self.simulated_deformed_node = None
        self.measured_initial_node = None
        self.measured_deformed_node = None

        self.simulated_initial_surfnode = None
        self.simulated_deformed_surfnode = None
        self.measured_initial_surfnode = None
        self.measured_deformed_surfnode = None

        self.simulated_initial_centroid = None
        self.simulated_deformed_centroid = None
        self.measured_initial_centroid = None
        self.measured_deformed_centroid = None

    def set_lesion_coord(self, simulated_initial = None, simulated_deformed = None, measured_initial = None, measured_deformed = None):
        if simulated_initial is not None:
            self.simulated_initial_node = np.array(simulated_initial)
        if simulated_deformed is not None:
            self.simulated_deformed_node = np.array(simulated_deformed)
        if measured_initial is not None:
            self.measured_initial_node = np.array(measured_initial)
        if measured_deformed is not None:
            self.measured_deformed_node = np.array(measured_deformed)
        
        self.calc_lesion_surface()
        self.calc_lesion_centroid(source="surfnode")

    def calc_lesion_surface(self, simulated: bool = True, measured: bool = True):
        
        if simulated:

            # find convex hull
            initial_hull = ConvexHull(self.simulated_initial_node)

            # query the hull points (points on hull surface)
            surf_idx = initial_hull.vertices
            self.simulated_initial_surfnode = self.simulated_initial_node[surf_idx]
            self.simulated_deformed_surfnode = self.simulated_deformed_node[surf_idx]

        if measured:

            # find convex hull
            measured_initial_hull = ConvexHull(self.measured_initial_node)
            measured_deformed_hull = ConvexHull(self.measured_deformed_node)

            # query the hull points (points on hull surface)
            surf_idx = measured_initial_hull.vertices
            self.measured_initial_surfnode = self.measured_initial_node[surf_idx]
            
            surf_idx = measured_deformed_hull.vertices
            self.measured_deformed_surfnode = self.measured_deformed_node[surf_idx]

    def calc_lesion_centroid(self, source: Literal["node", "surfnode"] = "surfnode"):

        if source == "node":
            simulated_initial_points = self.simulated_initial_node
            simulated_deformed_points = self.simulated_deformed_node
            measured_initial_points = self.measured_initial_node
            measured_deformed_points = self.measured_deformed_node
            
        elif source == "surfnode":
            self.calc_lesion_surface()
            simulated_initial_points = self.simulated_initial_surfnode
            simulated_deformed_points = self.simulated_deformed_surfnode
            measured_initial_points = self.measured_initial_surfnode
            measured_deformed_points = self.measured_deformed_surfnode 
        
        self.simulated_initial_centroid = np.mean(simulated_initial_points, axis=0)
        self.simulated_deformed_centroid = np.mean(simulated_deformed_points, axis=0)
        self.measured_initial_centroid = np.mean(measured_initial_points, axis=0)
        self.measured_deformed_centroid = np.mean(measured_deformed_points, axis=0)
 
    def calc_chamfer_distance(self, A, B):
        """
        Computes the chamfer distance between two sets of points A and B.
        """
        tree = KDTree(B)
        dist_A = tree.query(A)[0]
        tree = KDTree(A)
        dist_B = tree.query(B)[0]
        return np.mean(dist_A) + np.mean(dist_B)      
    
    def calc_rotation(self, P, Q):
        # Compute centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        # Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Compute covariance matrix
        H = np.dot(P_centered.T, Q_centered)
        
        # Perform SVD
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        return Rotation.from_matrix(R)
    
    def get_lesion_centroid(self):
        return  self.simulated_initial_centroid, self.simulated_deformed_centroid, \
                self.measured_initial_centroid, self.measured_deformed_centroid
    
    def get_lesion_volume_displacements(self, simulated: bool = True, measured: bool = True, deformed: bool = True):

        total_displacement = np.zeros((3, 3), dtype=float)
        total_magnitude = np.zeros((3,), dtype=float)
                
        for i in range(0, 3):
            
            # select the pair to calculate
            if i == 0 and simulated:
                p_true = self.simulated_initial_node
                p_pred = self.simulated_deformed_node
            elif i == 1 and measured:
                p_true = self.measured_initial_node
                p_pred = self.measured_deformed_node
            elif i == 2 and deformed:
                p_true = self.measured_deformed_node
                p_pred = self.simulated_deformed_node
                                        
            # calculate displacement
            if p_pred.shape == p_true.shape:
                nodal_displacements = p_pred - p_true
                nodal_magnitudes = np.linalg.norm(nodal_displacements, axis=1)
                total_displacement[i, :] = np.mean(nodal_displacements, axis=0)
                total_magnitude[i] = np.linalg.norm(total_displacement[i, :])
            else:
                total_displacement[i, :] = np.nan
                total_magnitude[i] = np.nan

        return total_displacement, total_magnitude

    def get_lesion_surface_displacements(self, simulated: bool = True, measured: bool = True, deformed: bool = True):
        
        total_displacement = np.zeros((3, 3), dtype=float)
        total_magnitude = np.zeros((3,), dtype=float)
                
        for i in range(0, 3):
            
            # select the pair to calculate
            if i == 0 and simulated:
                p_true = self.simulated_initial_surfnode
                p_pred = self.simulated_deformed_surfnode
            elif i == 1 and measured:
                p_true = self.measured_initial_surfnode
                p_pred = self.measured_deformed_surfnode
            elif i == 2 and deformed:
                p_true = self.measured_deformed_surfnode
                p_pred = self.simulated_deformed_surfnode
                                        
            # calculate displacement
            if p_pred.shape == p_true.shape:
                nodal_displacements = p_pred - p_true
                nodal_magnitudes = np.linalg.norm(nodal_displacements, axis=1)
                total_displacement[i, :] = np.mean(nodal_displacements, axis=0)
                total_magnitude[i] = np.linalg.norm(total_displacement[i, :])
            else:
                total_displacement[i, :] = np.nan
                total_magnitude[i] = np.nan

        return total_displacement, total_magnitude

    def get_lesion_centroid_displacements(self, simulated: bool = True, measured: bool = True, deformed: bool = True): 

        centroid_displacements = np.zeros((3, 3), dtype=float)
        centroid_magnitudes = np.zeros((3,), dtype=float)
                
        for i in range(0, 3):
            
            # select the pair to calculate
            if i == 0 and simulated:
                p_true = self.simulated_initial_centroid
                p_pred = self.simulated_deformed_centroid
            elif i == 1 and measured:
                p_true = self.measured_initial_centroid
                p_pred = self.measured_deformed_centroid
            elif i == 2 and deformed:
                p_true = self.measured_deformed_centroid
                p_pred = self.simulated_deformed_centroid
                                        
            # calculate displacement
            centroid_displacements[i, :] = p_pred - p_true
            centroid_magnitudes[i] = np.linalg.norm(centroid_displacements[i, :])

        return centroid_displacements, centroid_magnitudes
    
    def get_lesion_hausdorff_distance(self):
        
        hausdorff_distances = np.zeros((3,), dtype=float)
                
        for i in range(0, 3):
            
            # select the pair to calculate
            if i == 0:
                p_true = self.simulated_initial_surfnode
                p_pred = self.simulated_deformed_surfnode
            elif i == 1:
                p_true = self.measured_initial_surfnode
                p_pred = self.measured_deformed_surfnode
            elif i == 2:
                p_true = self.measured_deformed_surfnode
                p_pred = self.simulated_deformed_surfnode
            
            # by the preliminary test, calculation between the pair of node-node and surfnode-surfnode create equal results
            hausdorff_distances[i] = directed_hausdorff(p_pred, p_true)[0]
            
        return hausdorff_distances
    
    def get_lesion_chamfer_distance(self):
        
        chamfer_distances = np.zeros((3,), dtype=float)
                
        for i in range(0, 3):
            
            # select the pair to calculate
            if i == 0:
                p_true = self.simulated_initial_surfnode
                p_pred = self.simulated_deformed_surfnode
            elif i == 1:
                p_true = self.measured_initial_surfnode
                p_pred = self.measured_deformed_surfnode
            elif i == 2:
                p_true = self.measured_deformed_surfnode
                p_pred = self.simulated_deformed_surfnode
            
            # by the preliminary test, calculation between the pair of node-node and surfnode-surfnode create equal results
            chamfer_distances[i] = self.calc_chamfer_distance(p_pred, p_true)
            
        return chamfer_distances
        
    def get_simulated_lesion_surface_rotation(self):
        p = self.simulated_initial_surfnode
        q = self.simulated_deformed_surfnode
        r = self.calc_rotation(p, q)
        angles = r.as_euler("xyz", degrees=True)
        return angles  # rx, ry, rz in "xyz" rotation sequence
        

class Report:

    logger = TaskLogger("Report")

    def __init__(self) -> None:
        # data
        self.headers = ["Field", "X (mm)", "Y (mm)", "Z (mm)", " Magnitude (mm)"]
        self.tabular_data = np.empty((len(self.headers),), dtype=object)
        
        # tabulate arguments
        self.decimal_precision = 4
        self.tablefmt = "outline"
        
    def handle_inputs(self, initial, deformed):
        i = np.array(initial)
        d = np.array(deformed)
        return i, d
    
    def set_bounds(self, initial, deformed):

        handler = BoundsResultHandler()
        handler.set_bounds(initial, deformed)

        disp, mag = handler.get_displacements()
        
        # construct the data
        data_ni = np.array(["Initial Negative Bounds", initial[0], initial[2], initial[4], ""])
        data_nd = np.array(["Deformed Negative Bounds", deformed[0], deformed[2], deformed[4], ""])
        data_dn = np.array(["Negative Bounds Displacement", disp[0, 0], disp[0, 1], disp[0, 2], mag[0]])
        data_pi = np.array(["Initial Positive Bounds", initial[1], initial[3], initial[5], ""])
        data_pd = np.array(["Deformed Positive Bounds", deformed[1], deformed[3], deformed[5], ""])
        data_dp = np.array(["Positive Bounds Displacement", disp[1, 0], disp[1, 1], disp[1, 2], mag[1]])
        data_d = np.array(["Total Bounds Displacement", disp[2, 0], disp[2, 1], disp[2, 2], mag[2]])

        # add the data of this category to table's data
        data = np.vstack((data_ni, data_nd, data_dn, data_pi, data_pd, data_dp, data_d))
        self.tabular_data = np.vstack((self.tabular_data, data))

    def set_simulated_domains(self, initial, deformed, displacement):
        self.domains_result_handler = DomainsResultHandler()
        self.domains_result_handler.set_simulated_domains(initial, deformed, displacement)

        i_vol, d_vol, _, _ = self.domains_result_handler.get_simulated_domain_volumes()
        data_i_vol = np.array(["Simulated Initial Domain Volume", "", "", "", i_vol])
        data_d_vol = np.array(["Simulated Deformed Domain Volume", "", "", "", d_vol])
        data_vol = np.vstack((data_i_vol, data_d_vol))

        max_mag_u, mean_mag_u, stddev_mag_u = self.domains_result_handler.get_nodal_displacements()
        data_max_mag_u = np.array(["Max of Simulated Domain Displacement Magnitude", "", "", "", max_mag_u])
        data_mean_mag_u = np.array(["Mean of Simulated Domain Displacement Magnitude", "", "", "", mean_mag_u])
        data_stddev_mag_u = np.array(["S.D. of Simulated Domain Displacement Magnitude", "", "", "", stddev_mag_u])
        data_mag_u = np.vstack((data_max_mag_u, data_mean_mag_u, data_stddev_mag_u))
        
        dist, _, _ = self.domains_result_handler.get_hausdorff_distances()
        data_hd = np.array(["Hausdorff Distance between Simulated Domain", "", "", "", dist])

        self.tabular_data = np.vstack((self.tabular_data, data_vol, data_mag_u, data_hd))

    def set_measured_domains(self, initial, deformed):
        self.domains_result_handler.set_measured_domains(initial, deformed)
        
        _, _, i_vol, d_vol = self.domains_result_handler.get_simulated_domain_volumes()
        data_i_vol = np.array(["Measured Initial Domain Volume", "", "", "", i_vol])
        data_d_vol = np.array(["Measured Deformed Domain Volume", "", "", "", d_vol])
        data_vol = np.vstack((data_i_vol, data_d_vol))
        
        _, m_dist, d_dist = self.domains_result_handler.get_hausdorff_distances()
        data_m_hd = np.array(["Hausdorff Distance between Measured Domain", "", "", "", m_dist])
        data_d_hd = np.array(["Hausdorff Distance between Deformed Domain", "", "", "", d_dist])
        data_hd = np.vstack((data_m_hd, data_d_hd))
        
        self.tabular_data = np.vstack((self.tabular_data, data_vol, data_hd))
    
    def set_markers_coord(self, initial, deformed):
        # handle inputs
        i, d = self.handle_inputs(initial, deformed)
    
    def set_lesion_volume(self, simulated_initial, simulated_deformed):
        data_si = np.array(["Simulated Initial Lesions Volume", "", "", "", simulated_initial])
        data_sd = np.array(["Simulated Deformed Lesions Volume", "", "", "", simulated_deformed])
        data = np.vstack((data_si, data_sd)) 
        
        # stack every data
        self.tabular_data = np.vstack((self.tabular_data, data))

    def set_lesion_coord(self, simulated_initial, simulated_deformed, measured_initial, measured_deformed):

        handler = LesionResultHandler()
        handler.set_lesion_coord(simulated_initial, simulated_deformed, measured_initial, measured_deformed)

        disp, mag = handler.get_lesion_volume_displacements()
        data_s = np.array(["Simulated Lesion Volume Displacement", disp[0, 0], disp[0, 1], disp[0, 2], mag[0]])
        data_m = np.array(["Measured Lesion Volume Displacement", disp[1, 0], disp[1, 1], disp[1, 2], mag[1]])
        data_d = np.array(["Lesion Volume Displacement Error", disp[2, 0], disp[2, 1], disp[2, 2], mag[2]])
        data_dv = np.vstack((data_s, data_m, data_d))
        
        disp, mag = handler.get_lesion_surface_displacements()
        data_s = np.array(["Simulated Lesion Surface Displacement", disp[0, 0], disp[0, 1], disp[0, 2], mag[0]])
        data_m = np.array(["Measured Lesion Surface Displacement", disp[1, 0], disp[1, 1], disp[1, 2], mag[1]])
        data_d = np.array(["Lesion Surface Displacement Error", disp[2, 0], disp[2, 1], disp[2, 2], mag[2]])
        data_ds = np.vstack((data_s, data_m, data_d))
        
        disp, mag = handler.get_lesion_centroid_displacements()
        data_s = np.array(["Simulated Lesion Centroid Displacement", disp[0, 0], disp[0, 1], disp[0, 2], mag[0]])
        data_m = np.array(["Measured Lesion Centroid Displacement", disp[1, 0], disp[1, 1], disp[1, 2], mag[1]])
        data_d = np.array(["Lesion Centroid Displacement Error", disp[2, 0], disp[2, 1], disp[2, 2], mag[2]])
        data_dc = np.vstack((data_s, data_m, data_d))

        si, sd, mi, md = handler.get_lesion_centroid()
        data_si = np.array(["Simulated Initial Lesion Centroid", si[0], si[1], si[2], ""])
        data_sd = np.array(["Simulated Deformed Lesion Centroid", sd[0], sd[1], sd[2], ""])
        data_mi = np.array(["Measured Initial Lesion Centroid", mi[0], mi[1], mi[2], ""])
        data_md = np.array(["Measured Deformed Lesion Centroid", md[0], md[1], md[2], ""])
        data_cc = np.vstack((data_si, data_sd, data_mi, data_md))

        dist = handler.get_lesion_hausdorff_distance()
        data_s = np.array(["Hausdorff Distance between Simulated Lesions", "", "", "", dist[0]])
        data_m = np.array(["Hausdorff Distance between Measured Lesions", "", "", "", dist[1]])
        data_d = np.array(["Hausdorff Distance between Deformed Lesions", "", "", "", dist[2]])
        data_dh = np.vstack((data_s, data_m, data_d))

        dist = handler.get_lesion_chamfer_distance()
        data_s = np.array(["Chamfer Distance between Simulated Lesions", "", "", "", dist[0]])
        data_m = np.array(["Chamfer Distance between Measured Lesions", "", "", "", dist[1]])
        data_d = np.array(["Chamfer Distance between Deformed Lesions", "", "", "", dist[2]])
        data_dch = np.vstack((data_s, data_m, data_d))
        
        rx, ry, rz = handler.get_simulated_lesion_surface_rotation()
        data_a = np.array(["Simulated Lesion Surface XYZ Rotation (deg)", rx, ry, rz, ""])

        # stack every data
        self.tabular_data = np.vstack((self.tabular_data, data_dv, data_ds, data_cc, data_dc, data_dh, data_dch, data_a))

    def set_contact_residual_penetration(self, residual_penetration: float):
        data_p = np.array(["Contact Residual Penetration", "", "", "", residual_penetration])
        self.tabular_data = np.vstack((self.tabular_data, data_p))  

    def get_cleaned_tabular_data(self, tabular_data):
        # Iterate over each element in the array

        # iterate row
        for i in range(0, tabular_data.shape[0]):

            # iterate column, but not column 0 (field)
            for j in range(1, tabular_data.shape[1]):

                # remark 1: entries of multi-type numpy array have type 'str'
                # remark 2: empty string ("") equals False
                if isinstance(tabular_data[i, j], str) and tabular_data[i, j]:
                    
                    # clear "nan"
                    if tabular_data[i, j] == "nan":
                        tabular_data[i, j] = ""
                    
                    # round float
                    else:
                        tabular_data[i, j] = float(tabular_data[i, j])
                        tabular_data[i, j] = round(tabular_data[i, j], self.decimal_precision)

        return tabular_data[1:]  # neglect the empty row data (row 0)
   
    def get_report_table(self):
        self.tabular_data = self.get_cleaned_tabular_data(self.tabular_data)
        report_table = tabulate(
            tabular_data=self.tabular_data,
            headers=self.headers,
            tablefmt=self.tablefmt,
        )
        return "\n" + report_table

    def save_report(self, saved_directory: str = r""):
        os.makedirs(saved_directory, exist_ok=True)
        report_df = pd.DataFrame(
            columns=self.headers,
            data=self.tabular_data,
        )
        saved_path = saved_directory + r"error.csv"
        report_df.to_csv(saved_path, index=False)
        self.logger.loginfo(f"{saved_path} is successfully saved.")


class MeshPostAnalyzer:

    logger = TaskLogger("MeshPostAnalyzer")

    def __init__(self):

        # meshes from simulated result
        self.domain_initial = None
        self.domain_deformed = None
        self.embedded_domain_initial = None
        self.embedded_domain_deformed = None

        # numpy array indices from simulated result
        self.domain_elemidx_embedded_domain = None
        self.domain_nodeidx_fixed = None
        self.domain_nodeidx_submerged = None

        # added meshes from initial mesh and measured mesh
        self.collided_domain_initial = None
        self.measured_domain_initial = None
        self.measured_domain_deformed = None
        self.measured_embedded_domain_initial = None
        self.measured_embedded_domain_deformed = None

        # plotting instances
        self.plotter = None
        self.legend_entries = None
        self.fluid_level = None

        # tabular data instances
        self.headers = ["Feature", "Metrics", "X", "Y", "Z", "Magnitude", "Unit"]
        self.tabular_data = np.empty((len(self.headers),), dtype=object)
        self.decimal_precision = 4
        self.tablefmt = "outline"

    def read_components(self, directory: str):

        self.directory = directory

        self.domain_initial = pv.read(directory + r"domain_initial.msh")
        self.domain_deformed = pv.read(directory + r"domain_deformed.msh")
        self.embedded_domain_initial = pv.read(directory + r"embedded_domain_initial.msh")
        self.embedded_domain_deformed = pv.read(directory + r"embedded_domain_deformed.msh")

        self.domain_elemidx_embedded_domain = np.load(directory + r"domain_elemidx_embedded_domain.npy")
        self.domain_nodeidx_fixed = np.load(directory + r"domain_nodeidx_fixed.npy")

        try:
            self.domain_nodeidx_submerged = np.load(directory + r"domain_nodeidx_submerged.npy")
        except FileNotFoundError:
            pass

        total_displacement_coord = self.domain_deformed.points - self.domain_initial.points
        self.domain_deformed["Displacement"] = np.linalg.norm(total_displacement_coord, axis=1)

        # query surface mesh
        self.domainsurf_initial = self.domain_initial.extract_surface()
        self.domainsurf_deformed = self.domain_deformed.extract_surface()
        self.embedded_domainsurf_initial = self.embedded_domain_initial.extract_surface()
        self.embedded_domainsurf_deformed = self.embedded_domain_deformed.extract_surface()
        self.measured_embedded_domainsurf_initial = self.measured_embedded_domain_initial.extract_surface()
        self.measured_embedded_domainsurf_deformed = self.measured_embedded_domain_deformed.extract_surface()

        # query only external surface of measured domain mesh (2d)
        self.measured_domainextsurf_deformed = self.measured_domain_deformed.extract_largest().extract_surface()

        # query only external surface of simulated domain mesh (3d)
        tree_embedded_domainsurf_deformed = KDTree(self.embedded_domainsurf_deformed.points.astype(np.double))
        self.domainextsurf_elemidx = np.empty((0), dtype=int)  # indices of face elements
        for i, dc in enumerate(self.domainsurf_deformed.cell):
            dist, idx = tree_embedded_domainsurf_deformed.query(dc.points)
            if (dist == [0, 0, 0]).any(): continue
            self.domainextsurf_elemidx = np.append(self.domainextsurf_elemidx, i)
        self.domainextsurf_initial = self.domainsurf_initial.extract_cells(self.domainextsurf_elemidx)
        self.domainextsurf_deformed = self.domainsurf_deformed.extract_cells(self.domainextsurf_elemidx)
        self.measured_domainextsurf_initial = self.domainextsurf_initial
            
        # query mesh centroid
        self.domaincentroid_initial = np.mean(self.domainsurf_initial.points, axis=0)
        self.domaincentroid_deformed = np.mean(self.domainsurf_deformed.points, axis=0)
        self.embedded_domaincentroid_initial = np.mean(self.embedded_domainsurf_initial.points, axis=0)
        self.embedded_domaincentroid_deformed = np.mean(self.embedded_domainsurf_deformed.points, axis=0)
        self.measured_domaincentroid_initial = np.mean(self.measured_domainextsurf_initial.points, axis=0)
        self.measured_domaincentroid_deformed = np.mean(self.measured_domainextsurf_deformed.points, axis=0)
        self.measured_embedded_domaincentroid_initial = np.mean(self.measured_embedded_domainsurf_initial.points, axis=0)
        self.measured_embedded_domaincentroid_deformed = np.mean(self.measured_embedded_domainsurf_deformed.points, axis=0)
        
        # query only unfixed nodes
        domain_nodeidx_unfixed_mask = np.ones(self.domain_deformed.n_points, dtype=bool)
        domain_nodeidx_unfixed_mask[self.domain_nodeidx_fixed] = False
        self.domain_nodeidx_unfixed = np.where(domain_nodeidx_unfixed_mask)[0]

        # query nodeidx of embedded domain w.r.t. domain
        tree_domain_deformed = KDTree(self.domain_deformed.points.astype(np.double))
        dist, idx = tree_domain_deformed.query(self.embedded_domain_deformed.points)
        self.domain_nodeidx_embedded_domain = idx
        
    def add_collided_domain(self, directory: str = None, mesh: pv.UnstructuredGrid = None, goal_orientation: np.ndarray = np.array([0, 0, 0]), goal_position: np.ndarray = np.array([0, 0, 0])):
        if directory is not None and mesh is None:
            self.collided_domain_initial = pv.read(directory)

        elif directory is None and mesh is not None:
            self.collided_domain_initial = mesh
            
        self.collided_domain_initial.rotate_x(goal_orientation[0], inplace=True)
        self.collided_domain_initial.rotate_y(goal_orientation[1], inplace=True)
        self.collided_domain_initial.rotate_z(goal_orientation[2], inplace=True)
        self.collided_domain_initial.translate(goal_position, inplace=True)

    def add_measured_meshes(self, domain_initial, domain_deformed, embedded_domain_initial, embedded_domain_deformed, goal_orientation: np.ndarray = np.array([0, 0, 0])):
        self.measured_domain_initial = domain_initial
        self.measured_domain_deformed = domain_deformed
        self.measured_embedded_domain_initial = embedded_domain_initial
        self.measured_embedded_domain_deformed = embedded_domain_deformed
        meshes = [
            self.measured_domain_initial,
            self.measured_domain_deformed,
            self.measured_embedded_domain_initial,
            self.measured_embedded_domain_deformed
        ]
        for mesh in meshes:
            mesh.rotate_x(goal_orientation[0], inplace=True)
            mesh.rotate_y(goal_orientation[1], inplace=True)
            mesh.rotate_z(goal_orientation[2], inplace=True)

    def set_fluid_level(self, fluid_level: float = None):
        self.fluid_level = fluid_level
    
    def add_mesh_macros(self, configuration: Literal["initial", "deformed"], part: Literal["brain", "skull", "lesion", "lesion_measured", "lesion_nodes", "fluid", "lesion_displacements", "fixed_nodes", "submerged_nodes"], opacity: float = 1.0, wireframe: bool = False):
        
        color_dict = {
            "initial": {
                "skull": "LightGray",
                "brain": "LightPink",
                "lesion": "MediumTurquoise",
                "lesion_nodes": "MediumTurquoise",
                "lesion_displacements": "Gold",
                "fluid": "LightSkyBlue",
                "fixed_nodes": "Black",
                "submerged_nodes": "LightSkyBlue",
            },
            "deformed": {
                "skull": "LightGray",
                "brain": "Black",
                "lesion_measured": "YellowGreen",
                "lesion": "OrangeRed",
                "lesion_nodes": "OrangeRed",
                "lesion_displacements": "Gold",
                "fixed_nodes": "Black",
                "submerged_nodes": "LightSkyBlue",
            }
        }
        color = color_dict[configuration][part]

        if part == "brain":
            if configuration == "initial":
                self.plotter.add_mesh(self.domain_initial, color=color, opacity=opacity, lighting=True)
                if wireframe:
                    self.plotter.add_mesh(self.domain_initial, style="wireframe", color="Black", line_width=1, opacity=opacity)
                self.legend_entries.append(["Initial Brain Phantom", color])
            elif configuration == "deformed":
                sargs = dict(title_font_size=28, label_font_size=28, height=0.6, vertical=True, position_x=0.85, position_y=0.2)
                self.plotter.add_mesh(self.domain_deformed, show_edges=False, scalars="Displacement", cmap="viridis", lighting=True, smooth_shading=True, opacity=opacity, scalar_bar_args=sargs)
                if wireframe:
                    self.plotter.add_mesh(self.domain_deformed, style="wireframe", color="Black", line_width=1, opacity=opacity)
                self.legend_entries.append(["Deformed Brain Phantom", color])

        elif part == "fixed_nodes":
            if configuration == "initial":
                node_fixed = self.domain_initial.points[self.domain_nodeidx_fixed]
            elif configuration == "deformed":
                node_fixed = self.domain_deformed.points[self.domain_nodeidx_fixed]
            self.plotter.add_mesh(pv.PolyData(node_fixed), color=color, point_size=15, render_points_as_spheres=True)
            self.legend_entries.append(["Fixed Nodes", color])

        elif (configuration == "initial" or configuration == "deformed") and part == "skull":
            self.plotter.add_mesh(mesh=self.collided_domain_initial, color=color, lighting=True, opacity=opacity)
            if wireframe:
                self.plotter.add_mesh(mesh=self.collided_domain_initial, style="wireframe", color="Black", line_width=1, opacity=1)
            self.legend_entries.append(["Initial Skull Phantom", color])

        elif part == "lesion":
            if configuration == "initial":
                self.plotter.add_mesh(mesh=self.embedded_domain_initial, color=color, opacity=opacity)
                if wireframe:
                    self.plotter.add_mesh(mesh=self.embedded_domain_initial, style="wireframe", color="Black", line_width=1, opacity=1)
                self.legend_entries.append(["Simulated Initial Lesion", color])
            elif configuration == "deformed":
                self.plotter.add_mesh(mesh=self.embedded_domain_deformed, color=color, opacity=opacity)
                if wireframe:
                    self.plotter.add_mesh(mesh=self.embedded_domain_deformed, style="wireframe", color="Black", line_width=1, opacity=1)
                self.legend_entries.append(["Simulated Deformed Lesion", color])

        elif part == "lesion_measured":
            if configuration == "initial":
                self.plotter.add_mesh(mesh=self.measured_embedded_domain_initial, color=color, opacity=opacity)
                if wireframe:
                    self.plotter.add_mesh(mesh=self.measured_embedded_domain_initial, style="wireframe", color="Black", line_width=1, opacity=1)
                self.legend_entries.append(["Measured Initial Lesion", color])
            elif configuration == "deformed":
                self.plotter.add_mesh(mesh=self.measured_embedded_domain_deformed, label="Deformed Lesion", color=color, opacity=opacity)
                if wireframe:
                    self.plotter.add_mesh(mesh=self.measured_embedded_domain_deformed, style="wireframe", color="Black", line_width=1, opacity=1)
                self.legend_entries.append(["Measured Deformed Lesion", color])

        elif part == "lesion_nodes":
            if configuration == "initial":
                self.plotter.add_mesh(mesh=self.embedded_domain_initial.points, color=color, render_points_as_spheres=True, point_size=10)
                self.legend_entries.append(["Initial Lesion Nodes", color])
            elif configuration == "deformed":
                self.plotter.add_mesh(mesh=self.embedded_domain_deformed.points, color=color, render_points_as_spheres=True, point_size=10)
                self.legend_entries.append(["Deformed Lesion Nodes", color])

        elif configuration == "initial" and part == "fluid":
            if self.fluid_level is not None: 
                x_min, _, _, _, _, _ = self.domain_initial.bounds
                _, x_max, y_min, y_max, z_min, _ = self.collided_domain_initial.bounds
                margin = 10
                fluid_bounds =  [x_min - margin, x_max + margin, y_min - margin, y_max + margin, z_min, self.fluid_level]
                fluid_poly = pv.Box(bounds=fluid_bounds)
                self.plotter.add_mesh(mesh=fluid_poly, color=color, opacity=opacity)
                self.legend_entries.append(["Fluid", color])
        
        elif configuration == "deformed" and part == "lesion_displacements":
            for pi, pd in zip(self.embedded_domain_initial.points, self.embedded_domain_deformed.points):
                self.plotter.add_mesh(pv.Arrow(pi, pd - pi, scale="auto"), label="Simulated Lesion Displacement", color=color, render_lines_as_tubes=True, line_width=5)
            self.legend_entries.append(["Simulated Lesion Displacement", color])

        elif configuration == "deformed" and part == "submerged_nodes":
            if self.domain_nodeidx_submerged is not None:
                self.domain_nodeidx_submerged = np.unique(self.domain_nodeidx_submerged)
                self.plotter.add_mesh(self.domain_deformed.points[self.domain_nodeidx_submerged], label="Submerged Nodes", color=color, render_points_as_spheres=True, point_size=12)
                self.legend_entries.append(["Submerged Nodes", color])
  
    def calc_bounds(self, a: tuple, b: tuple):
        """calculate difference of bounds

        Parameters
        ----------
        a : tuple
            bounds of referenced or ground truth mesh, in the format of [xmin, xmax, ymin, ymax, zmin, zmax]
        b : tuple
            bounds of predicted or measured mesh, in the format of [xmin, xmax, ymin, ymax, zmin, zmax]

        Returns
        -------
        float
            difference of bounds, in the format of [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        
        a = np.array(a)
        b = np.array(b)

        i_lowerbnd = [0, 2, 4]
        i_upperbnd = [1, 3, 5]
        
        a_lowerbnd = a[i_lowerbnd]
        a_upperbnd = a[i_upperbnd]
        b_lowerbnd = b[i_lowerbnd]
        b_upperbnd = b[i_upperbnd]

        diff_lowerbnd = b_lowerbnd - a_lowerbnd
        diff_upperbnd = b_upperbnd - a_upperbnd
        
        pdiff_lowerbnd = (diff_lowerbnd / a_lowerbnd) * 100
        pdiff_upperbnd = (diff_upperbnd / a_upperbnd) * 100
        
        return diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd
        
    def calc_volume(self, a: pv.UnstructuredGrid, b: pv.UnstructuredGrid):
        av = a.volume
        bv = b.volume
        diffv = bv - av
        pdiffv = (diffv / av) * 100
        return av, bv, diffv, pdiffv

    def calc_radius(self, a: np.ndarray, b: np.ndarray):
        ca = np.mean(a, axis=0)
        cb = np.mean(b, axis=0)
        
        ra = np.linalg.norm(a - ca, axis=1)
        rb = np.linalg.norm(b - cb, axis=1)
        rdiff = rb - ra
        prdiff = (rdiff / ra) * 100
        
        ra_min = np.amin(ra)
        ra_max = np.amax(rb)
        ra_mean = np.mean(ra)
        ra_sd = np.std(ra)
        ra_data = np.array([ra_min, ra_max, ra_mean, ra_sd])
        
        rb_min = np.amin(rb)
        rb_max = np.amax(rb)
        rb_mean = np.mean(rb)
        rb_sd = np.std(rb)
        rb_data = np.array([rb_min, rb_max, rb_mean, rb_sd])
        
        rdiff_min = np.amin(rdiff)
        rdiff_max = np.amax(rdiff)
        rdiff_mean = np.mean(rdiff)
        rdiff_sd = np.std(rdiff)
        rdiff_data = np.array([rdiff_min, rdiff_max, rdiff_mean, rdiff_sd])
        
        prdiff_min = np.amin(prdiff)
        prdiff_max = np.amax(prdiff)
        prdiff_mean = np.mean(prdiff)
        prdiff_sd = np.std(prdiff)
        prdiff_data = np.array([prdiff_min, prdiff_max, prdiff_mean, prdiff_sd])
        
        return ra_data, rb_data, rdiff_data, prdiff_data
        
    def calc_displacement(self, a: np.ndarray, b: np.ndarray):
        
        # check pairwise
        if a.shape == b.shape:
            
            dispvects = b - a
            
            if a.ndim == 2 and b.ndim == 2:
                dispmags = np.linalg.norm(dispvects, axis=1)
                idx_min = np.argmin(dispmags)
                idx_max = np.argmax(dispmags)
                disp_min = dispvects[idx_min]
                disp_max = dispvects[idx_max]
                disp_mean = np.mean(dispvects, axis=0)            
                disp_sd = np.std(dispvects, axis=0)

            elif a.ndim == 1 and b.ndim == 1:
                dispmags = np.linalg.norm(dispvects, axis=0)
                disp_min = dispvects
                disp_max = dispvects
                disp_mean = dispvects
                disp_sd = np.array([0., 0., 0.])

        elif a.shape != b.shape:
            self.logger.loginfo("Skip calculating the 'displacements' because the shape of inputs are not equal.")
            disp_min = None
            disp_max = None
            disp_mean = None            
            disp_sd = None

        return disp_min, disp_max, disp_mean, disp_sd
    
    def calc_rotation(self, a: np.ndarray, b: np.ndarray):

        # check pairwise
        if a.shape == b.shape:
            
            # Compute centroids
            a_centroid = np.mean(a, axis=0)
            b_centroid = np.mean(b, axis=0)
            
            # Center the points
            a_centered = a - a_centroid
            b_centered = b - b_centroid
            
            # Compute covariance matrix
            H = np.dot(a_centered.T, b_centered)
            
            # Perform SVD
            U, _, Vt = np.linalg.svd(H)
            
            # Compute rotation matrix
            R = np.dot(Vt.T, U.T)
            
            # Handle reflection
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
        
            return Rotation.from_matrix(R)
        
        elif a.shape != b.shape:
            self.logger.loginfo("Skip calculating the 'rotation' because the shape of inputs are not equal.")
            return None

    def calc_euclidean_distance(self, a: np.ndarray, b: np.ndarray):
        # check pairwise
        if a.shape == b.shape:
            
            dispvects = b - a
            
            if a.ndim == 2 and b.ndim == 2:
                dispmags = np.linalg.norm(dispvects, axis=1)
                ed_min = np.amin(dispmags)
                ed_max = np.amax(dispmags)
                ed_mean = np.mean(dispmags)            
                ed_sd = np.std(dispmags)

            elif a.ndim == 1 and b.ndim == 1:
                dispmags = np.linalg.norm(dispvects, axis=0)
                ed_min = dispmags
                ed_max = dispmags
                ed_mean = dispmags
                ed_sd = np.array([0., 0., 0.])

        elif a.shape != b.shape:
            self.logger.loginfo("Skip calculating the 'euclidean distance' because the shape of inputs are not equal.")
            ed_min = None
            ed_max = None
            ed_mean = None            
            ed_sd = None

        return ed_min, ed_max, ed_mean, ed_sd

    def calc_hausdorff_distance(self, a: np.ndarray, b: np.ndarray):

        """calculate full hausdorff distance

        Parameters
        ---------
        a : np.ndarray
            surface of referenced or ground truth mesh

        b : np.ndarray
            surface of predicted or measured mesh

        Returns
        -------
        hd : float
            hausdorff distance between a and b mesh
        ia : int
            index of a mesh that determine the hausdorff distance
        ib : int
            index of b mesh that determine the hausdorff distance
        """

        hd_fwd = directed_hausdorff(a, b)  # forward hausdorff distance
        hd_rvs = directed_hausdorff(b, a)  # reverse hausdorff distance

        hds = np.vstack([hd_fwd, hd_rvs])
        ihd = np.argmax(hds[:, 0])
        hd = hds[ihd, :]  # full hausdorff distance (consider both direction)

        if ihd == 0:    # forward hausdorff distance is the hausdorff distance
            ia = hd[1]
            ib = hd[2]
        elif ihd == 1:  # reverse hausdorff distance is the hausdorff distance
            ia = hd[2]
            ib = hd[1]

        return hd[0], int(ia), int(ib)
    
    def calc_strains(self, a: pv.UnstructuredGrid, b: pv.UnstructuredGrid):
        calculator = StrainCalculator()

        # linearized strains
        linearized_strain_array_file_path = self.directory + r"linearized_strains.npy"
        linearized_strains_cell_data = None
        try:
            linearized_strains_cell_data = np.load(linearized_strain_array_file_path)
            self.logger.loginfo("Linearized strain array file is found, skip calculating the 'linearized strain'.")
        except FileNotFoundError:
            self.logger.loginfo("Linearized strain array file is not found, calculating the 'linearized strain'.")
            linearized_strains_cell_data = calculator.calc_linearized_strains(a, b)
            np.save(linearized_strain_array_file_path, linearized_strains_cell_data)
            self.logger.loginfo(f"Linearized strain array file is saved at {linearized_strain_array_file_path}.")

        # von mises strains
        von_mises_strain_array_file_path = self.directory + r"von_mises_strains.npy"
        von_mises_strains_cell_data = None
        try:
            von_mises_strains_cell_data = np.load(von_mises_strain_array_file_path)
            self.logger.loginfo("Von Mises strain array file is found, skip calculating the 'von mises strain'.")
        except FileNotFoundError:
            self.logger.loginfo("Von Mises strain array file is not found, calculating the 'von mises strain'.")
            von_mises_strains_cell_data = calculator.calc_strains(a, b)
            np.save(von_mises_strain_array_file_path, von_mises_strains_cell_data)
            self.logger.loginfo(f"Von Mises strain array file  is saved at {von_mises_strain_array_file_path}.")
            
        return linearized_strains_cell_data, von_mises_strains_cell_data

    def visualize(self, mode: Literal["visualize", "save_screenshot"] = "visualize", save_orbit: bool = False, saved_screenshot_root_directory: str = "", saved_graphic_root_directory: str = "", saved_orbiting_screenshot_root_directory: str = ""):

        # handle visualization flag
        visualize = True if mode == "visualize" else False
        save_screenshot = True if mode == "save_screenshot" else False

        # create pyvista plotter
        if visualize: is_off_screen = False
        if save_screenshot: is_off_screen = True
        
        # define constants for visualization and screenshots saving
        # variants = list(range(0, 15))
        # variants = list(range(0, 8))
        variants = list(range(8, 15))
        view_vectors = [
            VIEW_VECTORS["isometric_top"],
            VIEW_VECTORS["right"],
            VIEW_VECTORS["left"],
        ]
        view_configs = [0, 1, 2]
        wireframe_configs = [False, True]
        nodes_plotted_configs = [False, True]

        if visualize or save_screenshot:
            visualization_configs = itertools.product(variants, view_configs, wireframe_configs, nodes_plotted_configs)
            
            for i_img, (level, i_view, is_wireframe, is_nodes_plotted) in enumerate(visualization_configs):
                self.plotter = pv.Plotter(window_size=[960, 1024], off_screen=is_off_screen)
                self.legend_entries = []
                view = view_vectors[i_view]

                if level == 0:
                    # initial all opaque
                    self.add_mesh_macros("initial", "skull", 1, is_wireframe)
                    self.add_mesh_macros("initial", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("initial", "fixed_nodes")

                elif level == 1:
                    # initial brain opaque
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("initial", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("initial", "fixed_nodes")

                elif level == 2:
                    # initial lesion opaque
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("initial", "brain", 0.3)
                    self.add_mesh_macros("initial", "lesion", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("initial", "fixed_nodes")

                elif level == 3:
                    # initial lesion opaque with fluid
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("initial", "brain", 0.3)
                    self.add_mesh_macros("initial", "lesion", 1, is_wireframe)
                    self.add_mesh_macros("initial", "fluid", 0.2)
                    if is_nodes_plotted:
                        self.add_mesh_macros("initial", "fixed_nodes")

                elif level == 4:
                    # deformed all opaque
                    self.add_mesh_macros("initial", "skull", 1, is_wireframe)
                    self.add_mesh_macros("deformed", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("deformed", "submerged_nodes")
                        self.add_mesh_macros("deformed", "fixed_nodes")
                        
                elif level == 5:
                    # deformed brain opaque
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("deformed", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("deformed", "submerged_nodes")
                        self.add_mesh_macros("deformed", "fixed_nodes")

                elif level == 6:
                    # deformed lesion opaque
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("deformed", "brain", 0.3)
                    self.add_mesh_macros("deformed", "lesion", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("deformed", "submerged_nodes")
                        self.add_mesh_macros("deformed", "fixed_nodes")
                        
                elif level == 7:
                    # deformed lesion translucent with initial lesion
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("deformed", "brain", 0.3)
                    self.add_mesh_macros("initial", "lesion", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion", 1, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("deformed", "submerged_nodes")
                        self.add_mesh_macros("deformed", "fixed_nodes")

                elif level == 8:
                    # deformed lesion translucent with measured lesion
                    self.add_mesh_macros("initial", "skull", 0.05)
                    self.add_mesh_macros("deformed", "brain", 0.3)
                    self.add_mesh_macros("deformed", "lesion", 1, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        self.add_mesh_macros("deformed", "submerged_nodes")
                        self.add_mesh_macros("deformed", "fixed_nodes")
                        
                elif level == 9:
                    # lesion focus
                    self.add_mesh_macros("initial", "lesion", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        continue

                elif level == 10:
                    # lesion focus
                    self.add_mesh_macros("deformed", "lesion", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        continue

                elif level == 11:
                    # lesion focus
                    self.add_mesh_macros("deformed", "lesion_nodes")
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        continue

                elif level == 12:
                    # lesion focus
                    self.add_mesh_macros("initial", "lesion", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        continue

                elif level == 13:
                    # lesion focus
                    self.add_mesh_macros("initial", "lesion_nodes")
                    self.add_mesh_macros("deformed", "lesion_nodes")
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    if is_nodes_plotted:
                        continue
                
                elif level == 14:
                    # lesion focus
                    self.add_mesh_macros("initial", "lesion_nodes")
                    self.add_mesh_macros("deformed", "lesion_nodes")
                    self.add_mesh_macros("deformed", "lesion_measured", 0.5, is_wireframe)
                    self.add_mesh_macros("deformed", "lesion_displacements")
                    if is_nodes_plotted:
                        continue

                self.legend_entries, ind = np.unique(self.legend_entries, axis=0, return_index=True)
                self.legend_entries = self.legend_entries[np.argsort(ind)]
                # self.plotter.add_axes()
                # self.plotter.add_legend(labels=self.legend_entries.tolist(), bcolor=None)
                grid = self.plotter.show_grid()
                self.plotter.reset_camera()
                self.plotter.view_vector(view)
                self.plotter.camera.zoom(1.1)  # 1.2 for cuboidbrain3, 1.1 for mrbraintumor1
                self.plotter.show(auto_close=False)
                
                if save_screenshot:
                    # make directory
                    os.makedirs(saved_screenshot_root_directory, exist_ok=True)
                    os.makedirs(saved_graphic_root_directory, exist_ok=True)

                    # file name
                    if i_view == 0: view_char = "i"
                    elif i_view == 1: view_char = "l"
                    elif i_view == 2: view_char = "r"
                    is_wireframe_char = "w" if is_wireframe else "x"
                    is_nodes_plotted_char = "n" if is_nodes_plotted else "x"
                    screenshot_file_name = f"img{level:02}_{view_char}_{is_wireframe_char}{is_nodes_plotted_char}"

                    # save screenshot images (*.png)
                    screenshot_path = saved_screenshot_root_directory + f"{screenshot_file_name}.png"
                    self.plotter.screenshot(filename=screenshot_path)
                    self.logger.loginfo(f"The screenshot is saved at {screenshot_path}")
                    
                    # save graphics (*.eps)
                    graphic_path = saved_graphic_root_directory + f"{screenshot_file_name}.eps"
                    self.plotter.save_graphic(filename=graphic_path)
                    self.logger.loginfo(f"The graphic    is saved at {graphic_path}")
                    
                if save_orbit and level in range(0, 9) and i_view == 2:
                    # make directory
                    os.makedirs(saved_orbiting_screenshot_root_directory, exist_ok=True)
                    
                    # save orbiting screenshot images (*.gif / *.mp4)
                    view_char = "o"
                    is_wireframe_char = "w" if is_wireframe else "x"
                    is_nodes_plotted_char = "n" if is_nodes_plotted else "x"
                    orbiting_file_name = f"img{level:02}_{view_char}_{is_wireframe_char}{is_nodes_plotted_char}"
                    orbiting_gif_path = saved_orbiting_screenshot_root_directory + f"{orbiting_file_name}.gif"
                    orbiting_mp4_path = saved_orbiting_screenshot_root_directory + f"{orbiting_file_name}.mp4"
                    
                    path = self.plotter.generate_orbital_path(n_points=72, shift=0)
                    
                    try:
                        self.plotter.remove_actor(grid)
                        self.plotter.remove_legend()
                        self.plotter.remove_scalar_bar()
                    except:
                        pass
                    
                    self.plotter.open_gif(orbiting_gif_path)
                    self.plotter.orbit_on_path(path, write_frames=True)
                    self.plotter.open_movie(orbiting_mp4_path)
                    self.plotter.orbit_on_path(path, write_frames=True)
                    self.plotter.close()
                    self.logger.loginfo(f"The orbiting image is saved at {orbiting_gif_path}")
                    self.logger.loginfo(f"The orbiting movie is saved at {orbiting_mp4_path}")

                self.plotter.clear()

    def result_analysis(self, saved_result_root_directory: str = "", register_to_sra: np.ndarray = np.array([0, 0, 0])):

        # """
        # * bounds of domains (simulated)
        a = self.domain_initial.bounds
        b = self.domain_deformed.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Simulated Domains", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Simulated Domains", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Simulated Domains", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Simulated Domains", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")

        # * bounds of domains (measured)
        a = self.measured_domain_initial.bounds
        b = self.measured_domain_deformed.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Measured Domains", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Measured Domains", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Measured Domains", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Measured Domains", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")

        # * bounds of domains (error)
        a = self.measured_domain_deformed.bounds
        b = self.domain_deformed.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Domains Error", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Domains Error", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Domains Error", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Domains Error", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")

        # * volume of domains (simulated)
        a = self.domain_initial
        b = self.domain_deformed
        av, bv, diffv, pdiffv = self.calc_volume(a, b)
        self.add_data_to_report("Simulated Domain (Initial)", "Volume", mag=av, unit="mm^3")
        self.add_data_to_report("Simulated Domain (Deformed)", "Volume", mag=bv, unit="mm^3")
        self.add_data_to_report("Simulated Domains", "Change in Volume", mag=diffv, unit="mm^3")
        self.add_data_to_report("Simulated Domains", "%Change in Volume", mag=pdiffv, unit="%")

        # * displacement of domains (simulated)
        a = self.domain_initial.points[self.domain_nodeidx_unfixed]
        b = self.domain_deformed.points[self.domain_nodeidx_unfixed]
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Simulated Domains", "Displacement", vmin=disp_min, vmax=disp_max, vmean=disp_mean, vsd=disp_sd, unit="mm")

        # * displacement of domains centroid (simulated)
        a = self.domaincentroid_initial
        b = self.domaincentroid_deformed
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Simulated Domains", "Centroid Displacement", mag=disp_mean, unit="mm")
        
        # * euclidean distance of domains (simulated)
        a = self.domain_initial.points[self.domain_nodeidx_unfixed]
        b = self.domain_deformed.points[self.domain_nodeidx_unfixed]
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Simulated Domains", "Euclidean Distance", vmin=ed_min, vmax=ed_max, vmean=ed_mean, vsd=ed_sd, unit="mm")

        # * euclidean distance of domains centroid (simulated)
        a = self.domaincentroid_initial
        b = self.domaincentroid_deformed
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Simulated Domains", "Centroid Euclidean Distance", mag=ed_mean, unit="mm")
        
        # * hausdorff distance of domains (simulated)
        a = self.domainextsurf_initial.points
        b = self.domainextsurf_deformed.points
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Simulated Domains", "Hausdorff Distance", mag=hd, unit="mm")
        
        # * hausdorff distance of domains (measured)
        a = self.measured_domainextsurf_initial.points
        b = self.measured_domainextsurf_deformed.points
        x_interested_threshold = -5
        a = a[a[:, 0] > x_interested_threshold]
        b = b[b[:, 0] > x_interested_threshold]
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Measured Domains", "Hausdorff Distance", mag=hd, unit="mm")

        # * hausdorff distance of domains (error)
        a = self.measured_domainextsurf_deformed.points
        b = self.domainextsurf_deformed.points
        x_interested_threshold = -5
        a = a[a[:, 0] > x_interested_threshold]
        b = b[b[:, 0] > x_interested_threshold]
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Domains Error", "Hausdorff Distance", mag=hd, unit="mm")

        # * bounds of translation-excluded embedded domain (simulated)
        a = self.embedded_domain_initial.translate(-1 * self.embedded_domaincentroid_initial)
        b = self.embedded_domain_deformed.translate(-1 * self.embedded_domaincentroid_deformed)
        a = a.bounds
        b = b.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")

        # * bounds of translation-excluded embedded domain (measured)
        a = self.measured_embedded_domain_initial.translate(-1 * self.measured_embedded_domaincentroid_initial)
        b = self.measured_embedded_domain_deformed.translate(-1 * self.measured_embedded_domaincentroid_deformed)
        a = a.bounds
        b = b.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Measured Embedded Domains", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Measured Embedded Domains", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Measured Embedded Domains", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Measured Embedded Domains", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")
        
        # * bounds of translation-excluded embedded domain (error)
        a = self.measured_embedded_domain_deformed.translate(-1 * self.measured_embedded_domaincentroid_deformed)
        b = self.embedded_domain_deformed.translate(-1 * self.embedded_domaincentroid_deformed)
        a = a.bounds
        b = b.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Embedded Domains Error", "Change in Negative Bounds", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Embedded Domains Error", "Change in Positive Bounds", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Embedded Domains Error", "%Change in Negative Bounds", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Embedded Domains Error", "%Change in Positive Bounds", mag=pdiff_upperbnd, unit="%")

        # * volume of embedded domain (simulated)
        a = self.embedded_domain_initial
        b = self.embedded_domain_deformed
        av, bv, diffv, pdiffv = self.calc_volume(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Volume", mag=av, unit="mm^3")
        self.add_data_to_report("Simulated Embedded Domains", "Volume", mag=bv, unit="mm^3")
        self.add_data_to_report("Simulated Embedded Domains", "Change in Volume", mag=diffv, unit="mm^3")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Volume", mag=pdiffv, unit="%")
        
        # * radius of embedded domain (simulated)
        a = self.embedded_domainsurf_initial.points
        b = self.embedded_domainsurf_deformed.points
        ra_data, rb_data, rdiff_data, prdiff_data = self.calc_radius(a, b)
        self.add_data_to_report("Simulated Embedded Domain (Initial)", "Radius", mag=ra_data, unit="mm")
        self.add_data_to_report("Simulated Embedded Domain (Deformed) ", "Radius", mag=rb_data, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "Change  in Radius", mag=rdiff_data, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Radius", mag=prdiff_data, unit="%")
                
        # * displacement of embedded domains (simulated)
        a = self.embedded_domain_initial.points
        b = self.embedded_domain_deformed.points
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Displacement", vmin=disp_min, vmax=disp_max, vmean=disp_mean, vsd=disp_sd, unit="mm")

        # * displacement of embedded domains centroid (simulated)
        a = self.embedded_domaincentroid_initial
        b = self.embedded_domaincentroid_deformed
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Centroid Displacement", mag=disp_mean, unit="mm")
        
        # * displacement of embedded domains centroid (measured)
        a = self.measured_embedded_domaincentroid_initial
        b = self.measured_embedded_domaincentroid_deformed
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Measured Embedded Domains", "Centroid Displacement", mag=disp_mean, unit="mm")
        
        # * displacement of embedded domains centroid (error)
        a = self.measured_embedded_domaincentroid_deformed
        b = self.embedded_domaincentroid_deformed
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Embedded Domains Error", "Centroid Displacement", mag=disp_mean, unit="mm")
        
        # * displacement of embedded domains centroid (registered) (simulated)
        a = self.embedded_domaincentroid_initial
        a = pv.PolyData(a)
        a = a.rotate_x(register_to_sra[0], inplace=False)
        a = a.rotate_y(register_to_sra[1], inplace=False)
        a = a.rotate_z(register_to_sra[2], inplace=False)
        a = a.points
        b = self.embedded_domaincentroid_deformed
        b = pv.PolyData(b)
        b = b.rotate_x(register_to_sra[0], inplace=False)
        b = b.rotate_y(register_to_sra[1], inplace=False)
        b = b.rotate_z(register_to_sra[2], inplace=False)
        b = b.points
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Centroid Displacement (Reg.) (SRA)", mag=disp_mean, unit="mm")
        
        # * displacement of embedded domains centroid (registered) (measured)
        a = self.measured_embedded_domaincentroid_initial
        a = pv.PolyData(a)
        a = a.rotate_x(register_to_sra[0], inplace=False)
        a = a.rotate_y(register_to_sra[1], inplace=False)
        a = a.rotate_z(register_to_sra[2], inplace=False)
        a = a.points
        b = self.measured_embedded_domaincentroid_deformed
        b = pv.PolyData(b)
        b = b.rotate_x(register_to_sra[0], inplace=False)
        b = b.rotate_y(register_to_sra[1], inplace=False)
        b = b.rotate_z(register_to_sra[2], inplace=False)
        b = b.points
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Measured Embedded Domains", "Centroid Displacement (Reg.) (SRA)", mag=disp_mean, unit="mm")
        
        # * displacement of embedded domains centroid (registered) (error)
        a = self.measured_embedded_domaincentroid_deformed
        a = pv.PolyData(a)
        a = a.rotate_x(register_to_sra[0], inplace=False)
        a = a.rotate_y(register_to_sra[1], inplace=False)
        a = a.rotate_z(register_to_sra[2], inplace=False)
        a = a.points
        b = self.embedded_domaincentroid_deformed
        b = pv.PolyData(b)
        b = b.rotate_x(register_to_sra[0], inplace=False)
        b = b.rotate_y(register_to_sra[1], inplace=False)
        b = b.rotate_z(register_to_sra[2], inplace=False)
        b = b.points
        disp_min, disp_max, disp_mean, disp_sd = self.calc_displacement(a, b)
        self.add_data_to_report("Embedded Domains Error", "Centroid Displacement (Reg.) (SRA)", mag=disp_mean, unit="mm")
        
        # * euler xyz rotation of embedded domains (simulated)
        a = self.embedded_domainsurf_initial.points
        b = self.embedded_domainsurf_deformed.points
        r = self.calc_rotation(a, b)
        eulxyz_ab = r.as_euler(seq="xyz", degrees=True)
        eulxyz_ba = r.inv().as_euler(seq="xyz", degrees=True)
        self.add_data_to_report("Simulated Embedded Domains", "Euler XYZ Rotation", mag=eulxyz_ab, unit="deg")

        # * bounds of transformed-excluded embedded domain (simulated)
        a = self.embedded_domain_initial.translate(-1 * self.embedded_domaincentroid_initial)
        b = self.embedded_domain_deformed
        b = b.translate(-1 * self.embedded_domaincentroid_deformed)
        b = b.rotate_x(eulxyz_ba[0], inplace=False)
        b = b.rotate_y(eulxyz_ba[1], inplace=False)
        b = b.rotate_z(eulxyz_ba[2], inplace=False)
        a = a.bounds
        b = b.bounds
        diff_lowerbnd, diff_upperbnd, pdiff_lowerbnd, pdiff_upperbnd = self.calc_bounds(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Change in Negative Bounds (Reg.)", mag=diff_lowerbnd, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "Change in Positive Bounds (Reg.)", mag=diff_upperbnd, unit="mm")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Negative Bounds (Reg.)", mag=pdiff_lowerbnd, unit="%")
        self.add_data_to_report("Simulated Embedded Domains", "%Change in Positive Bounds (Reg.)", mag=pdiff_upperbnd, unit="%")
        
        # * euclidean distance of embedded domains (simulated)
        a = self.embedded_domain_initial.points
        b = self.embedded_domain_deformed.points
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Euclidean Distance", vmin=ed_min, vmax=ed_max, vmean=ed_mean, vsd=ed_sd, unit="mm")

        # * euclidean distance of embedded domains centroid (simulated)
        a = self.embedded_domaincentroid_initial
        b = self.embedded_domaincentroid_deformed
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Simulated Embedded Domains ", "Centroid Euclidean Distance", mag=ed_mean, unit="mm")
        
        # * euclidean distance of embedded domains centroid (simulated)
        a = self.measured_embedded_domaincentroid_initial
        b = self.measured_embedded_domaincentroid_deformed
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Measured Embedded Domains ", "Centroid Euclidean Distance", mag=ed_mean, unit="mm")

        # * euclidean distance of embedded domains centroid (simulated)
        a = self.measured_embedded_domaincentroid_deformed
        b = self.embedded_domaincentroid_deformed
        ed_min, ed_max, ed_mean, ed_sd = self.calc_euclidean_distance(a, b)
        self.add_data_to_report("Embedded Domains Error", "Centroid Euclidean Distance", mag=ed_mean, unit="mm")
        
        # * hausdorff distance of embedded domains (simulated)
        a = self.embedded_domainsurf_initial.points
        b = self.embedded_domainsurf_deformed.points
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Simulated Embedded Domains", "Hausdorff Distance", mag=hd, unit="mm")
        
        # * hausdorff distance of embedded domains (measured)
        a = self.measured_embedded_domain_initial.points
        b = self.measured_embedded_domain_deformed.points
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Measured Embedded Domains", "Hausdorff Distance", mag=hd, unit="mm")
        
        # * hausdorff distance of embedded domains (error)
        a = self.measured_embedded_domain_deformed.points
        b = self.embedded_domainsurf_deformed.subdivide(4, "linear").points
        hd, ia, ib = self.calc_hausdorff_distance(a, b)
        self.add_data_to_report("Embedded Domains Error", "Hausdorff Distance", mag=hd, unit="mm")
        # """        

        report = self.get_report_table()
        self.logger.loginfo(report)
        self.save_report(saved_result_root_directory)
        
        # * strains of simulated domains
        a = self.domain_initial
        b = self.domain_deformed
        linearized_strains_cell_data, von_mises_strains_cell_data = self.calc_strains(a, b)

    def strains_analysis(self):
        a = self.domain_initial
        b = self.domain_deformed
        linearized_strains_cell_data, von_mises_strains_cell_data = self.calc_strains(a, b)
        self.domain_deformed["Linearized Strain"] = linearized_strains_cell_data
        self.domain_deformed["Von Mises Strain"] = von_mises_strains_cell_data

        clipped = self.domain_deformed.clip(normal="y")
        self.plotter = pv.Plotter()
        self.legend_entries = []
        # self.plotter.add_mesh(self.domain_deformed, scalars="Linearized Strain", cmap="viridis", clim=[0.01, 0.05], below_color='blue', above_color='red', opacity=0.5)
        self.plotter.add_mesh(clipped, scalars="Linearized Strain", cmap="viridis", opacity=1, smooth_shading=True)
        self.plotter.show()

    def equilibrium_residual_analysis(self, visualize: bool = True, save_screenshot: bool = False, saved_screenshot_root_directory: str = "", saved_graphic_root_directory: str = ""):

        # load sparse array (K, F) and dense array (U)
        K = load_npz(self.directory + r"global_stiffness_matrix.npz")
        F = load_npz(self.directory + r"global_external_load_vector.npz")
        U = np.load(self.directory + r"global_displacement_vector.npy")

        # convert scipy.sparse.csr_matrix and np.ndarray to scipy.sparse.csr_array
        K = csr_array(K)
        F = csr_array(F)
        U = csr_array(U)

        # calculate residual vector
        R = (K @ U) - F
        R = R.toarray()
        R = R.reshape((-1, 3))

        # reshape and find magnitude for visualization
        # since the fixed nodes are not equilibrium (high residual), we neglect it by consider only unfixed nodes
        # the residual vector is first initialize in zeros vector, and overwrite only unfixed nodes
        r_vector_domain = np.zeros((self.domain_deformed.n_points, 3), dtype=float)
        r_vector_domain[self.domain_nodeidx_unfixed] = R[self.domain_nodeidx_unfixed]
        r_scalar_domain = np.linalg.norm(r_vector_domain, axis=1)
        
        # set vector and scalar to mesh
        self.domain_deformed["Equilibrium Residual Vector"] = r_vector_domain
        self.domain_deformed["Equilibrium Residual"] = r_scalar_domain
        self.domain_deformed.set_active_vectors("Equilibrium Residual Vector")
        self.embedded_domain_deformed["Equilibrium Residual Vector"] = r_vector_domain[self.domain_nodeidx_embedded_domain]
        self.embedded_domain_deformed["Equilibrium Residual"] = r_scalar_domain[self.domain_nodeidx_embedded_domain]
        self.embedded_domain_deformed.set_active_vectors("Equilibrium Residual Vector")

        # create pyvista plotter
        if visualize: is_off_screen = False
        if save_screenshot: is_off_screen = True
        
        # define constants for visualization and screenshots saving
        variants = list(range(0, 2))
        view_vectors = [
            VIEW_VECTORS["isometric_top"],
            VIEW_VECTORS["right"],
            VIEW_VECTORS["left"],
        ]
        view_configs = [0, 1, 2]
        
        # visualize
        if visualize or save_screenshot:
            visualization_configs = itertools.product(variants, view_configs)
            
            for i_img, (level, i_view) in enumerate(visualization_configs):
                self.plotter = pv.Plotter(window_size=[960, 1024], off_screen=is_off_screen)
                view = view_vectors[i_view]
                sargs = dict(title_font_size=28, label_font_size=28, height=0.6, vertical=True, position_x=0.85, position_y=0.2)
                
                if level == 0:
                    self.plotter.add_mesh(self.domain_deformed, scalars="Equilibrium Residual", cmap="plasma", lighting=True, smooth_shading=True, opacity=0.5, scalar_bar_args=sargs)
                    # self.plotter.add_mesh(self.domain_deformed.arrows, cmap="plasma", lighting=True, smooth_shading=True, opacity=1, scalar_bar_args=sargs)

                elif level == 1:
                    self.plotter.add_mesh(self.embedded_domain_deformed, scalars="Equilibrium Residual", cmap="plasma", lighting=True, smooth_shading=True, opacity=0.5, scalar_bar_args=sargs)
                    # self.plotter.add_mesh(self.embedded_domain_deformed.arrows, cmap="plasma", lighting=True, smooth_shading=True, opacity=1, scalar_bar_args=sargs)

                self.plotter.add_axes()
                self.plotter.show_grid()
                self.plotter.reset_camera()
                self.plotter.view_vector(view)
                self.plotter.camera.zoom(1.1)  # 1.2 for cuboidbrain3, 1.1 for mrbraintumor1
                self.plotter.show(auto_close=False)

                if save_screenshot:
                    # make directory
                    os.makedirs(saved_screenshot_root_directory, exist_ok=True)
                    os.makedirs(saved_graphic_root_directory, exist_ok=True)

                    # file name
                    if i_view == 0: view_char = "i"
                    elif i_view == 1: view_char = "l"
                    elif i_view == 2: view_char = "r"
                    numbering_offset = 15
                    level += numbering_offset                    
                    screenshot_file_name = f"img{level:02}_{view_char}"

                    # save screenshot images (*.png)
                    screenshot_path = saved_screenshot_root_directory + f"{screenshot_file_name}.png"
                    self.plotter.screenshot(filename=screenshot_path)
                    self.logger.loginfo(f"The screenshot is saved at {screenshot_path}")
                    
                    # save graphics (*.eps)
                    graphic_path = saved_graphic_root_directory + f"{screenshot_file_name}.eps"
                    self.plotter.save_graphic(filename=graphic_path)
                    self.logger.loginfo(f"The graphic    is saved at {graphic_path}")

                self.plotter.clear()
    
    def add_data_to_report(self, feature: str, metric: str, x = None, y = None, z = None, mag = None, vmin = None, vmax = None, vmean = None, vsd = None, unit: str = ""):
        
        def stack_datarow():
            datarow = np.array([feature, fieldrow, x, y, z, mag, unit])
            self.tabular_data = np.vstack((self.tabular_data, datarow))
    
        x = "" if x is None else x
        y = "" if y is None else y
        z = "" if z is None else z
        mag = "" if mag is None else mag
        vmin = "" if vmin is None else vmin
        vmax = "" if vmax is None else vmax
        vmean = "" if vmean is None else vmean
        vsd = "" if vsd is None else vsd
        
        if x or y or z:
            fieldrow = metric
            stack_datarow()

        if isinstance(mag, (float, np.ndarray)):
            fieldrow = metric
            if isinstance(mag, float):
                mag = mag
                stack_datarow()
            elif isinstance(mag, np.ndarray):
                if mag.size == 3:
                    x, y, z = mag
                    mag = ""
                    stack_datarow()
                elif mag.size == 4:
                    vmin, vmax, vmean, vsd = mag
                    mag = ""
            
        if isinstance(vmin, (float, np.ndarray)):
            fieldrow = metric + " (Min)"
            if isinstance(vmin, float): mag = vmin
            elif isinstance(vmin, np.ndarray) and vmin.size == 3: x, y, z = vmin
            stack_datarow()

        if isinstance(vmax, (float, np.ndarray)):
            fieldrow = metric + " (Max)"
            if isinstance(vmax, float): mag = vmax
            elif isinstance(vmax, np.ndarray) and vmax.size == 3: x, y, z = vmax
            stack_datarow()

        if isinstance(vmean, (float, np.ndarray)):
            fieldrow = metric + " (Mean)"
            if isinstance(vmean, float): mag = vmean
            elif isinstance(vmean, np.ndarray) and vmean.size == 3: x, y, z = vmean
            stack_datarow()

        if isinstance(vsd, (float, np.ndarray)):
            fieldrow = metric + " (S.D.)"
            if isinstance(vsd, float): mag = vsd
            elif isinstance(vsd, np.ndarray) and vsd.size == 3: x, y, z = vsd
            stack_datarow()

    def get_cleaned_tabular_data(self, tabular_data):
        # Iterate over each element in the array

        # iterate row
        for i in range(0, tabular_data.shape[0]):

            # iterate column, but not the first two columns (domain, metrics) and the last column (unit)
            for j in range(2, tabular_data.shape[1] - 1):

                # remark 1: entries of multi-type numpy array have type 'str'
                # remark 2: empty string ("") equals False
                if isinstance(tabular_data[i, j], str) and tabular_data[i, j]:
                    
                    # clear "nan"
                    if tabular_data[i, j] == "nan":
                        tabular_data[i, j] = ""
                    
                    # round float
                    else:
                        tabular_data[i, j] = float(tabular_data[i, j])
                        tabular_data[i, j] = round(tabular_data[i, j], self.decimal_precision)

        return tabular_data[1:]  # neglect the empty row data (row 0)
   
    def get_report_table(self):
        self.tabular_data = self.get_cleaned_tabular_data(self.tabular_data)
        report_table = tabulate(
            tabular_data=self.tabular_data,
            headers=self.headers,
            tablefmt=self.tablefmt,
        )
        return "\n" + report_table

    def save_report(self, saved_directory: str = r""):
        os.makedirs(saved_directory, exist_ok=True)
        report_df = pd.DataFrame(columns=self.headers, data=self.tabular_data)
        saved_path = saved_directory + r"result_analysis.csv"
        report_df.to_csv(saved_path, index=False)
        self.logger.loginfo(f"{saved_path} is successfully saved.")


class StrainCalculator:
    def __init__(self):
        pass

    def calc_linearized_strains(self, a: pv.UnstructuredGrid, b: pv.UnstructuredGrid):
        
        dispvects = b.points - a.points
        
        n_elem = b.n_cells
        total_strain_cell_data = np.zeros((n_elem,), dtype=float)
        n_tets = b.cells_dict[pv.CellType.TETRA].shape[0]
        total_strain = np.zeros((n_tets,), dtype=float)
    
        elements_progressbar = tqdm(
            b.cells_dict[pv.CellType.TETRA],
            desc="Linearized Strain Calculation",
            ncols=80,
            disable=False
        )
        
        for elemidx, nodeidx in enumerate(elements_progressbar):
            
            nodes_coord = b.points[nodeidx]
            elem_volume = self.__tetrahedral_volume(nodes_coord)
            dispvects_local = dispvects[nodeidx].flatten()
                        
            element_strain_matrix = self.__element_strain_matrix(nodes_coord, elem_volume)

            element_strain_vector = element_strain_matrix @ dispvects_local
            
            # Extract normal strains (ε_x, ε_y, ε_z)
            normal_strains = element_strain_vector[:3]  # First 3 components are normal strains
            
            # Extract shear strains (γ_xy, γ_yz, γ_zx)
            shear_strains = element_strain_vector[3:]  # Last 3 components are shear strains
            
            # Calculate the von Mises equivalent strain
            epsilon_x, epsilon_y, epsilon_z = normal_strains
            gamma_xy, gamma_yz, gamma_zx = shear_strains
            
            total_strain[elemidx] = np.sqrt(
                0.5 * (
                    (epsilon_x - epsilon_y)**2 + 
                    (epsilon_y - epsilon_z)**2 + 
                    (epsilon_z - epsilon_x)**2 + 
                    6 * (gamma_xy**2 + gamma_yz**2 + gamma_zx**2)
                )
            )
            
        # assign the strain value
        tets_mask = (b.celltypes == pv.CellType.TETRA)
        total_strain_cell_data[tets_mask] = total_strain
                        
        return total_strain_cell_data
                       
    def __tetrahedral_volume(self, nodes_coord: np.ndarray):
        """
        Calculate tetrahedral volume (referenced from chapter 9)

        Parameters
        ----------
        nodes_coord : np.ndarray
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        # insert column 0 with value of 1 into nodes_coord
        M = np.insert(nodes_coord, 0, 1, axis=1)

        # calculate tetrahedral volume
        V = np.abs((np.linalg.det(M)) / 6)

        return V
    
    def __element_strain_matrix(self, nodes_coord: np.ndarray, element_volume: float):
               
        # initialize matrix entries
        a, b, c, d = np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))
       
        # cyclic permutation
        cyclic_permutation_pool = np.array([1, 2, 3, 4])
        for index in range(0, 4):
            # define cyclic permutation index
            i = cyclic_permutation_pool[index]
            j = cyclic_permutation_pool[index - 3]
            k = cyclic_permutation_pool[index - 2]
            l = cyclic_permutation_pool[index - 1]
    
            # retrieve coord matrix for a, b, c, d calculation
            A = np.array([nodes_coord[j - 1], nodes_coord[k - 1], nodes_coord[l - 1]])
            B = copy.deepcopy(A)
            B[:, 0] = 1
            C = copy.deepcopy(A)
            C[:, 1] = 1
            D = copy.deepcopy(A)
            D[:, 2] = 1
            
            # a, b, c, d calculation; compensate index values to be 1, 2, 3, 4 instead 0, 1, 2, 3
            a[index] = np.power(-1, (j + k + l + 1)) * np.linalg.det(A)
            b[index] = np.power(-1, (j + k + l)) * np.linalg.det(B)
            c[index] = np.power(-1, (j + k + l)) * np.linalg.det(C)
            d[index] = np.power(-1, (j + k + l)) * np.linalg.det(D)

        # assign values into M
        M = np.zeros((6, 12))
        M[0, 0] = M[3, 1] = M[5, 2] = b[0]
        M[0, 3] = M[3, 4] = M[5, 5] = b[1]
        M[0, 6] = M[3, 7] = M[5, 8] = b[2]
        M[0, 9] = M[3, 10] = M[5, 11] = b[3]
        M[1, 1] = M[3, 0] = M[4, 2] = c[0]
        M[1, 4] = M[3, 3] = M[4, 5] = c[1]
        M[1, 7] = M[3, 6] = M[4, 8] = c[2]
        M[1, 10] = M[3, 9] = M[4, 11] = c[3]
        M[2, 2] = M[4, 1] = M[5, 0] = d[0]
        M[2, 5] = M[4, 4] = M[5, 3] = d[1]
        M[2, 8] = M[4, 7] = M[5, 6] = d[2]
        M[2, 11] = M[4, 10] = M[5, 9] = d[3]

        # calculate Be
        Be = M / (6 * element_volume)

        return Be
    
    def compute_jacobian(self, X):
        """
        Compute the Jacobian matrix J for a tetrahedral element in the reference configuration.
        Args:
            X: ndarray (4, 3), nodal coordinates of the reference tetrahedral element.
        Returns:
            J: ndarray (3, 3), Jacobian matrix.
        """
        # Compute edge vectors from node 0 to nodes 1, 2, and 3
        J = np.column_stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]))
        return J

    def compute_deformation_gradient(self, X, x):
        """
        Compute the deformation gradient F for a tetrahedral element.
        Args:
            X: ndarray (4, 3), reference nodal coordinates of the tetrahedral element.
            x: ndarray (4, 3), deformed nodal coordinates of the tetrahedral element.
        Returns:
            F: ndarray (3, 3), deformation gradient.
        """
        # Compute the Jacobian for the reference configuration
        J_ref = self.compute_jacobian(X)
        # Compute the Jacobian for the deformed configuration
        J_def = self.compute_jacobian(x)
        
        # Deformation gradient F        
        F = J_def @ np.linalg.inv(J_ref)
        return F           
            
    def compute_green_lagrange_strain(self, F):
        """
        Compute the Green-Lagrange strain tensor for a single element.
        Args:
            F: ndarray (3, 3), deformation gradient.
        Returns:
            E: ndarray (3, 3), Green-Lagrange strain tensor.
        """
        I = np.eye(3)  # Identity matrix
        C = F.T @ F    # Right Cauchy-Green deformation tensor
        E = 0.5 * (C - I)  # Green-Lagrange strain tensor
        return E
            
    def compute_principal_strains(self, E):
        """
        Compute the principal strains (eigenvalues of the strain tensor).
        Args:
            E: ndarray (3, 3), Green-Lagrange strain tensor.
        Returns:
            principal_strains: ndarray (3,), principal strains.
        """
        principal_strains, _ = np.linalg.eigh(E)
        return principal_strains

    def compute_von_mises_strain(self, E):
        """
        Compute the von Mises strain from the strain tensor.
        Args:
            E: ndarray (3, 3), Green-Lagrange strain tensor.
        Returns:
            von_mises: float, von Mises strain.
        """
        principal_strains = self.compute_principal_strains(E)
        von_mises = np.sqrt((2 / 3) * np.sum(principal_strains**2))
        return von_mises
    
    def compute_frobenius_norm(self, E):
        """
        Compute the Frobenius norm of the strain tensor.
        Args:
            E: ndarray (3, 3), Green-Lagrange strain tensor.
        Returns:
            frobenius_norm: float, Frobenius norm of the strain tensor.
        """
        return np.sqrt(np.sum(E**2))
     
    def calc_strains(self, a: pv.UnstructuredGrid, b: pv.UnstructuredGrid):
            
        n_elem = b.n_cells
        strains_cell_data = np.zeros((n_elem,), dtype=float)
        
        n_tets = b.cells_dict[pv.CellType.TETRA].shape[0]
        strains = np.zeros((n_tets,), dtype=float)
    
        # assign the strain value
        tets_mask = (b.celltypes == pv.CellType.TETRA)
        tetidx = np.where(tets_mask)[0]

        elements_progressbar = tqdm(
            tetidx,
            desc="Von Mises Strain Calculation",
            ncols=80,
            disable=False
        )
        
        for i, elemidx in enumerate(elements_progressbar):
            atet = a.extract_cells(elemidx)
            atetnode = atet.points
            btet = b.extract_cells(elemidx)
            btetnode = btet.points
            F = self.compute_deformation_gradient(atetnode, btetnode)
            strain_tensor = self.compute_green_lagrange_strain(F)
            strains[i] = self.compute_von_mises_strain(strain_tensor)

        strains_cell_data[tets_mask] = strains
                
        return strains_cell_data