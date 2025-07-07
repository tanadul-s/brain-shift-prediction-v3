import json
import glob
from typing import Literal

import numpy as np
import pyvista as pv

from src.core.fem_structural_analysis import SegmentedDomain
from src.utils.utils import parameters, constants, consoleconfigs, VIEW_VECTORS
from src.utils.loggerold import CustomLogger, TimeElapsedLog

# configs
LOG_LEVEL = consoleconfigs(config="logging_level")
USE_PROGRESSBAR = False


class Marker:
    def __init__(self, file_path: str) -> None:
        
        # attributes
        self.coordsys = None
        self.ras = None
        self.lps = None
        
        # load file *.json
        file = open(file_path, "r")
        self.data = json.load(file)
        
        # query
        self.coordsys, self.ras, self.lps = self.query()
        
    def query(self):
        # finding the coordinate system of the exported marker files
        coordsys = self.data["markups"][0]["coordinateSystem"]
        position = np.array(self.data["markups"][0]["controlPoints"][0]["position"])
        if coordsys == "LPS":
            lps = position
            ras = position * np.array([-1, -1, 1])
        elif coordsys == "RAS":
            ras = position
            lps = position * np.array([-1, -1, 1])
        return coordsys, ras, lps
            

class Markers(Marker):
    def __init__(self, file_path: str):
        self.marker_name_list = [
            "marker_refframe_o", "marker_refframe_x", "marker_refframe_y", "marker_refframe_z",
            "marker_float_1", "marker_float_2", "marker_float_3", "marker_float_4",
            "marker_lesion_centroid"
        ]
        super().__init__(file_path)

        
    def query(self):
        # finding the coordinate system of the exported marker files
        coordsys = self.data["markups"][0]["coordinateSystem"]
        position_dict_key = ["o", "x", "y", "z", "f1", "f2", "f3", "f4", "lc"]
        lps = [None] * len(position_dict_key)
        ras = [None] * len(position_dict_key)
        for i, name in enumerate(self.marker_name_list):
            label = np.array(self.data["markups"][0]["controlPoints"][i]["label"])
            position = np.array(self.data["markups"][0]["controlPoints"][i]["position"])
            if label == name:
                if coordsys == "LPS":
                    lps[i] = position
                    ras[i] = position * np.array([-1, -1, 1])
                elif coordsys == "RAS":
                    ras[i] = position
                    lps[i] = position * np.array([-1, -1, 1])
        
        position_lps_dict = dict(zip(position_dict_key, lps))
        position_ras_dict = dict(zip(position_dict_key, ras))
        
        return coordsys, position_lps_dict, position_ras_dict
        
        
class MarkersFixedFloatHandler:
    
    logger = CustomLogger(name="MKRS", level=LOG_LEVEL)
    etime_logger = CustomLogger(name="MKRS", level=LOG_LEVEL)
    
    def __init__(self, root_path: str, ref_fixed_coord_1: np.ndarray, ref_fixed_coord_2: np.ndarray, state: str = Literal["initial", "deformed"]) -> None:
        
        # handle inputs
        self.root_path = root_path
        self.ref_fixed_coord_1 = ref_fixed_coord_1
        self.ref_fixed_coord_2 = ref_fixed_coord_2
        self.state = state
        
        # list *.mkr.json files
        self.n_markers = 6  # 2 fixed marker and 4 float markers
        marker_fixed_file_list = glob.glob(rf"marker_{self.state}_fixed_*.mrk.json", root_dir=root_path)
        marker_float_file_list = glob.glob(rf"marker_{self.state}_float_*.mrk.json", root_dir=root_path)
        marker_file_list = [root_path + p for p in np.append(marker_fixed_file_list, marker_float_file_list)]
        
        # query marker's info from *.mrk.json files
        self.markers = np.array([None] * self.n_markers)
        self.markers_coord = np.zeros((self.n_markers, 3), dtype=np.double)
        self.markers_poly = None
        self.translated_markers = np.zeros((self.n_markers, 3), dtype=np.double)
        for i, p in enumerate(marker_file_list):
            self.markers[i] = Marker(p)
            self.markers_coord[i] = self.markers[i].lps
        self.markers_poly = pv.PolyData(self.markers_coord)
        
    def __marker_fixed_1_transformation(self, markers: pv.PolyData | pv.UnstructuredGrid, goal_orientation: np.ndarray = np.array([0.0, 0.0, 0.0])):
                
        # step 1:   `marker_fixed_1` will be exactly transformed into `ref_fixed_coord_1`,
        
        # firstly, rotate the marker system around the `marker_fixed_1`
        # to obtain the correct remapping between LPS and XYZ,
        # which depends on the object placement during CT scanning
        self.marker_fixed_1_coord_raw = self.markers_coord[0]  # untransformed coordinate of fixed marker
        markers.rotate_x(angle=goal_orientation[0], point=self.marker_fixed_1_coord_raw, inplace=True)
        markers.rotate_y(angle=goal_orientation[1], point=self.marker_fixed_1_coord_raw, inplace=True)
        markers.rotate_z(angle=goal_orientation[2], point=self.marker_fixed_1_coord_raw, inplace=True)
        
        # then, translate the reoriented marker system
        # by registering the `marker_fixed_1` into `ref_fixed_coord_1`
        self.lps_to_refxyz_translation_vector = self.ref_fixed_coord_1 - self.marker_fixed_1_coord_raw
        markers.translate(self.lps_to_refxyz_translation_vector, inplace=True)
        
        # update markers_poly
        if isinstance(markers, pv.PolyData):
            self.markers_poly = markers
            
        self.logger.info(f"\n\tFixed Marker 1 Orientation Transformation: {goal_orientation}")
        self.logger.info(f"\n\tFixed Marker 1 Translation Transformation: {self.lps_to_refxyz_translation_vector}")
        
        return markers
             
    def __marker_fixed_2_transformation(self, markers: pv.PolyData | pv.UnstructuredGrid):
        
        def two_vectors_to_rotation_matrix(v_measured, v_true):
            
            v1 = v_measured / np.linalg.norm(v_measured)
            v2 = v_true / np.linalg.norm(v_true)
            c = np.cross(v1, v2)
            d = np.dot(v1, v2)
            angle = np.arccos(d)
            
            # calculate rotation matrix
            skew_symmetric = np.array([
                [    0,  -c[2],   c[1]],
                [ c[2],      0,  -c[0]],
                [-c[1],   c[0],      0],
            ])
            rotation_matrix = np.eye(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * (1 - np.cos(angle)) / (angle ** 2)
            
            return rotation_matrix
        
        def rotation_matrix_to_euler_angles(rotation_matrix):
            
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            euler_angles = np.degrees([x, y, z])
            
            return euler_angles
            
        # step 2:   `marker_fixed_2` will be transformed into `ref_fixed_coord_2` in the closest position, and
        
        if isinstance(markers, pv.PolyData):
            
            # step 2.1: calculate rotation matrix from two vectors
            v_measured = self.markers_poly.points[1] - self.ref_fixed_coord_1  # vector from CT scan
            v_true = self.ref_fixed_coord_2 - self.ref_fixed_coord_1  # vector of simulation world
            rotation_matrix = two_vectors_to_rotation_matrix(v_measured, v_true)
            
            # step 2.2: calculate Euler angles from rotation matrix
            self.fine_registration_euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
            
        # step 2.3:
        markers.rotate_x(angle=self.fine_registration_euler_angles[0], point=self.ref_fixed_coord_1, inplace=True)
        markers.rotate_y(angle=self.fine_registration_euler_angles[1], point=self.ref_fixed_coord_1, inplace=True)
        markers.rotate_z(angle=self.fine_registration_euler_angles[2], point=self.ref_fixed_coord_1, inplace=True)

        self.logger.info(f"\n\tFixed Marker 2 Orientation Transformation: {self.fine_registration_euler_angles}")

        return markers

    def get_transformed_markers_poly(self, goal_orientation: np.ndarray = np.array([0.0, 0.0, 0.0])):
        
        # the ultimate goal is to transform all real markers' coordinates
        # into known coordinate of simulation world which can be obtained from CAD
        # where marker_fixed_1 is the main point that will be the reference of the marker system
        # and marker_fixed_2 will be the helper to align the orientation the the marker system
        # that is,
        # step 1:   `marker_fixed_1` will be exactly transformed into `ref_fixed_coord_1`,
        # step 2:   `marker_fixed_2` will be transformed into `ref_fixed_coord_2` in the closest position, and
        # (result): `marker_float_*` will be accordingly adjusted by the transformation of the `marker_fixed_*`

        self.markers_poly = self.__marker_fixed_1_transformation(self.markers_poly, goal_orientation)
        self.markers_poly = self.__marker_fixed_2_transformation(self.markers_poly)
        
        return self.markers_poly
    
    def get_transformed_markers_mesh(self, mesh_root_path: str, goal_orientation: np.ndarray = np.array([0.0, 0.0, 0.0])):
        
        # the ultimate goal is to transform all real markers' coordinates
        # into known coordinate of simulation world which can be obtained from CAD
        # where marker_fixed_1 is the main point that will be the reference of the marker system
        # and marker_fixed_2 will be the helper to align the orientation the the marker system
        # that is,
        # step 1:   `marker_fixed_1` will be exactly transformed into `ref_fixed_coord_1`,
        # step 2:   `marker_fixed_2` will be transformed into `ref_fixed_coord_2` in the closest position, and
        # (result): `marker_float_*` will be accordingly adjusted by the transformation of the `marker_fixed_*`
        
        # read mesh
        self.markers_mesh = pv.read(mesh_root_path + rf"cuboidbrain1_markers_{self.state}.msh")
        
        # transform from lps coordinate to referenced xyz coordinate, where is the position of fixed markers
        self.markers_mesh = self.__marker_fixed_1_transformation(self.markers_mesh, goal_orientation)
        self.markers_mesh = self.__marker_fixed_2_transformation(self.markers_mesh)
        
        return self.markers_mesh


class MarkersRefframeFloatLesionHandler:
    
    logger = CustomLogger(name="MKRS", level=LOG_LEVEL)
    etime_logger = CustomLogger(name="MKRS", level=LOG_LEVEL)
    
    def __init__(
            self, 
            initial_markers_root_path: str,
            initial_mesh_root_path: str,
            initial_refframe_dict: dict,
            initial_markers_file_name: str,
            deformed_markers_root_path: str,
            deformed_mesh_root_path: str,
            deformed_refframe_dict: dict,
            deformed_markers_file_name: str,
            segmented_brain: SegmentedDomain,
            segmented_markers: SegmentedDomain,
            segmented_lesion: SegmentedDomain,
        ) -> None:

        # handle inputs
        self.initial_markers_root_path = initial_markers_root_path
        self.initial_mesh_root_path = initial_mesh_root_path
        self.initial_refframe_dict = initial_refframe_dict
        self.deformed_markers_root_path = deformed_markers_root_path
        self.deformed_mesh_root_path = deformed_mesh_root_path
        self.deformed_refframe_dict = deformed_refframe_dict
        self.segmented_brain = segmented_brain
        self.segmented_markers = segmented_markers
        self.segmented_lesion = segmented_lesion
        
        # instants
        self.rotation_matrix = None
                
        # list *.mkr.json files
        initial_markers_file_path = initial_markers_root_path + rf"{initial_markers_file_name}.mrk.json"
        deformed_markers_file_path = deformed_markers_root_path + rf"{deformed_markers_file_name}.mrk.json"
        
        # query marker's info from *.mrk.json files
        initial_markers = Markers(initial_markers_file_path)
        self.initial_markers_dict = initial_markers.lps
        self.initial_markers_array = np.stack(list(self.initial_markers_dict.values()))
        self.initial_markers_poly = pv.PolyData(self.initial_markers_array)
        
        deformed_markers = Markers(deformed_markers_file_path)
        self.deformed_markers_dict = deformed_markers.lps
        self.deformed_markers_array = np.stack(list(self.deformed_markers_dict.values()))
        self.deformed_markers_poly = pv.PolyData(self.deformed_markers_array)
        
        # construct data in to list
        self.meshes = [
            self.initial_markers_poly,      
            self.segmented_brain.initial,   
            self.segmented_markers.initial, 
            self.segmented_lesion.initial,
            self.deformed_markers_poly,
            self.segmented_brain.deformed,
            self.segmented_markers.deformed,
            self.segmented_lesion.deformed,
        ]
                        
    def registration(self, domain: Literal["initial", "deformed"] = "initial"):
        
        # step 0: select domain
        if domain == "initial":
            destination_dict = self.initial_refframe_dict
            moving_dict = self.initial_markers_dict
            moving_poly = self.initial_markers_poly
        elif domain == "deformed":
            destination_dict = self.deformed_refframe_dict
            moving_dict = self.deformed_markers_dict
            moving_poly = self.deformed_markers_poly
        
        # step 1: coarse registration (fixed rotation and translation for origin)
        
        # initial fixed transform (rotation)
        rotate_modallps_to_cadxyz = np.array([90, 0, 90])
        self.apply_transformation(domain=domain, rotate=rotate_modallps_to_cadxyz, rotate_about_point=moving_dict["o"])
        
        # initial fixed transform (translation): the origin is aligned
        translate_modalxyz_to_cadxyz = destination_dict["o"] - moving_dict["o"]
        self.apply_transformation(domain=domain, translate=translate_modalxyz_to_cadxyz)
        
        
        # step 2: fine registration (rotation via SVD and translation  for origin)
        
        # calculate the rotation matrix between two point set
        fixed_points = np.stack(list(destination_dict.values()))[1:4]  # true refframe xyz
        moving_points = moving_poly.points[1:4]  # measured refframe xyz
        self.rotation_matrix = self.calculate_rotation_matrix(fixed_points, moving_points)
        
        # apply the rotation to the points
        transformation_matrix = self.construct_transformation_matrix(R=self.rotation_matrix)
        self.apply_transformation(domain=domain, transform=transformation_matrix)

        # post fixed transform (translation): the origin is aligned
        translate_modalxyz_to_cadxyz = destination_dict["o"] - moving_poly.points[0]
        self.apply_transformation(domain=domain, translate=translate_modalxyz_to_cadxyz)
                    
    def calculate_rotation_matrix(self, source_points, target_points):
        # Step 1: Remove the origin points, since they are already aligned
        source_axes = source_points # Remaining points along x, y, z axes
        target_axes = target_points # Remaining points along x, y, z axes

        # Step 2: Compute the covariance matrix
        H = np.dot(source_axes.T, target_axes)

        # Step 3: Perform SVD on the covariance matrix
        U, S, Vt = np.linalg.svd(H)

        # Step 4: Compute the rotation matrix
        R = np.dot(Vt.T, U.T)

        # Step 5: Ensure a proper rotation by checking the determinant
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        return R
    
    def construct_transformation_matrix(self, t: np.ndarray = np.zeros((3,), dtype=float), R: np.ndarray = np.zeros((3, 3)), dtype=float):
       # Create the 4x4 transformation matrix
        H = np.eye(4, dtype=float)  # Start with an identity matrix
        H[:3, :3] = R  # Insert the rotation matrix
        H[:3, 3] = t   # Insert the translation vector
        return H
     
    def apply_transformation(self, domain: Literal["initial", "deformed"] = "initial", rotate: np.ndarray = None, rotate_about_point: np.ndarray = np.array([0, 0, 0]), translate: np.ndarray = None, transform: np.ndarray = None):
        
        if domain == "initial":
            meshes = self.meshes[:4]
        elif domain == "deformed":
            meshes = self.meshes[4:]
        elif domain == "both":
            meshes = self.meshes
            
        for mesh in meshes:
            if rotate is not None:
                mesh.rotate_x(angle=rotate[0], point=rotate_about_point, inplace=True)
                mesh.rotate_y(angle=rotate[1], point=rotate_about_point, inplace=True)
                mesh.rotate_z(angle=rotate[2], point=rotate_about_point, inplace=True)
            if translate is not None:
                mesh.translate(xyz=translate, inplace=True)
            if transform is not None:
                mesh.transform(trans=transform, inplace=True)

    def get_measured_markers(self):
        return self.initial_markers_poly, self.measured_markers_mesh
    
    def get_rotation_matrix(self):
        return self.rotation_matrix
    
    def get_registered_measured_markers(self):
        p = self.initial_markers_poly.points
        registered_measured_markers = {
            "o": p[0],
            "x": p[1],
            "y": p[2],
            "z": p[3],
            "f1": p[4],
            "f2": p[5],
            "f3": p[6],
            "f4": p[7],
            "lc": p[8],
        }
        return self.initial_markers_poly, registered_measured_markers
    
    def get_registered_measured_markers_mesh(self):
        return self.measured_markers_mesh, self.measured_lesion_mesh, self.measured_brain_mesh
    
    def set_global_transformation(self, domain: Literal["initial", "deformed", "both"] = "initial", rotate: np.ndarray = None, rotate_about_point: np.ndarray = np.array([0, 0, 0]), translate: np.ndarray = None, transform: np.ndarray = None):
        self.apply_transformation(
            domain=domain,
            rotate=rotate,
            rotate_about_point=rotate_about_point,
            translate=translate,
            transform=transform
        )