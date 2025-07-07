import os
from typing import Literal
import copy
import itertools
import multiprocessing

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import pyvista as pv
from tqdm import tqdm

from src.utils.utils import parameters, constants, consoleconfigs, VIEW_VECTORS, sliced_matrix
from src.utils.result_utils import Report
from src.utils.logger import TaskLogger
from src.core.contact_mechanics import ContactMechanics, SelfContactMechanics
from src.solver.sparse_solver import LUDecompositionSolver, ConjugateGradientSolver


"""
<Phantom>
┃
┗━━━━━━ *.initial / *.deformed: <pv.UnstructuredGrid>
        ┗━━━━━━ *.points
        ┗━━━━━━ *.cells
        ┗━━━━━━ *.extract_surface()
        ┗━━━━━━ *.surface_indices()
        ┗━━━━━━ *.n_points
        ┗━━━━━━ *.n_cells
        ┗━━━━━━ *.bounds
"""

# constants
NODE_PER_ELEM = 4
DOF_PER_NODE = 3
DOF_PER_ELEM = NODE_PER_ELEM * DOF_PER_NODE
GRAVITATIONAL_ACCELERATION = constants(category="common", constant="gravitational_acceleration")

# configs
LOG_LEVEL = consoleconfigs(config="logging_level")
USE_PROGRESSBAR = consoleconfigs(config="use_progressbar")



class Domain:
    
    def __init__(self, directory: str = None, grid = None):
        
        self.directory = directory
        self.grid = grid
        
        if self.directory is not None and self.grid is None:
            if self.directory.lower().endswith((".msh")):
                self.initial = pv.read(filename=directory)
                self.deformed = None
            else:
                raise ValueError("File extension must be Gmsh's mesh or `*.msh`.")
        
        elif self.directory is None and self.grid is not None:
            self.initial = self.grid
            self.deformed = None
    
    def __del__(self):
        pass
        
    def set_orientation(self, goal_orientation: np.ndarray):
        self.initial.rotate_x(goal_orientation[0], inplace=True)
        self.initial.rotate_y(goal_orientation[1], inplace=True)
        self.initial.rotate_z(goal_orientation[2], inplace=True)
        
    def set_position(self, goal_position: np.ndarray):
        self.initial.translate(goal_position, inplace=True)

    def get_volume(self, domain: Literal["initial", "deformed"], cell_indices: np.ndarray = None):
        
        if domain == "initial": domain = self.initial
        elif domain == "deformed": domain = self.deformed
        
        cell_sizes = domain.compute_cell_sizes() 
        cell_volumes = cell_sizes.cell_data["Volume"]
        if cell_indices is None: mesh_volume = np.sum(cell_volumes)
        elif cell_indices is not None: mesh_volume = np.sum(cell_volumes[cell_indices])

        return mesh_volume


class ScannedDomain(Domain):
    
    def __init__(self, directory: str):
        super().__init__(directory=directory)


class EmbeddedDomain:
    
    def __init__(self, directory_initial: str = None, directory_deformed: str = None, grid_initial: pv.UnstructuredGrid = None, grid_deformed: pv.UnstructuredGrid = None):
        
        self.directory_initial = directory_initial
        self.directory_deformed = directory_deformed
        self.grid_initial = grid_initial
        self.grid_deformed = grid_deformed
        
        self.initial = None
        self.deformed = None
        
        if self.directory_initial is not None:
            if self.directory_initial.lower().endswith(".msh"):
                self.initial = pv.read(filename=self.directory_initial)
            else:
                raise ValueError("File extension must be Gmsh's mesh or `*.msh`.")
            
        if self.directory_deformed is not None:
            if self.directory_deformed.lower().endswith(".msh"):
                self.deformed = pv.read(filename=self.directory_deformed)
            else:
                raise ValueError("File extension must be Gmsh's mesh or `*.msh`.")
            
        if self.grid_initial is not None:
            self.initial = self.grid_initial
            
        if self.grid_deformed is not None:
            self.deformed = self.grid_deformed
    
    def __del__(self):
        pass
        
    def set_orientation(self, goal_orientation: np.ndarray):
        for m in [self.initial, self.deformed]:
            m.rotate_x(goal_orientation[0], inplace=True)
            m.rotate_y(goal_orientation[1], inplace=True)
            m.rotate_z(goal_orientation[2], inplace=True)
        
    def set_position(self, goal_position: np.ndarray):
        for m in [self.initial, self.deformed]:
            m.translate(goal_position, inplace=True)


class SegmentedDomain(EmbeddedDomain):
    def __init__(self, directory_initial = None, directory_deformed = None, grid_initial = None, grid_deformed = None):
        super().__init__(directory_initial, directory_deformed, grid_initial, grid_deformed)


class LinearElasticModel:
    
    logger = TaskLogger("LinearElasticModel")
    
    def __init__(self, domain: Domain):
        
        # constants
        self.YOUNGS_MODULUS = constants(category="silicone_mechanical_properties", constant="youngs_modulus")
        self.POISSONS_RATIO = constants(category="silicone_mechanical_properties", constant="poissons_ratio")
        self.MASS_DENSITY = constants(category="silicone_mechanical_properties", constant="mass_density")
        
        self.domain = domain
        self.GLOBAL_DOF = DOF_PER_NODE * self.domain.initial.n_points
        self.global_stiffness_matrix = None
        self.global_mass_matrix = None
        self.global_external_load_vector = None
        self.is_secondary_material_set = False

    def __del__(self):
        pass
    
    def __isotropic_material_constant_matrix(self):
        """
        Construct matrix of isotropic material constant (referenced from chapter 2)
        
        c: matrix of material constant
        c_common: common factor of entry c11 and c12
        c11: entries in matrix of material constant
        c12: entries in matrix of material constant
        G: entries in matrix of material constant (shear modulus)
        """
        
        c = np.zeros((6, 6))
        c_common = self.YOUNGS_MODULUS / ((1 - (2 * self.POISSONS_RATIO)) * (1 + self.POISSONS_RATIO))
        c11 = c_common * (1 - self.POISSONS_RATIO)
        c12 = c_common * self.POISSONS_RATIO
        G = (c11 - c12) / 2
    
        c[0, 0] = c[1, 1] = c[2, 2] = c11
        c[0, 1] = c[0, 2] = c[1, 2] = c[1, 0] = c[2, 0] = c[2, 1] = c12
        c[3, 3] = c[4, 4] = c[5, 5] = G
        
        return c
  
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
         
    def __element_stiffness_matrix(self, nodes_coord: np.ndarray, element_volume: float):
        c = self.__isotropic_material_constant_matrix()
        Be = self.__element_strain_matrix(nodes_coord, element_volume)
        Ke = element_volume * (Be.T @ c @ Be)
        return Ke
    
    def __element_mass_matrix(self, element_volume: float):
        # define symmetric matrix
        M = np.zeros((12, 12))
        for index in range(12):
            M[index, index] = 2
            M[index - 9, index] = M[index - 6, index] = M[index - 3, index] = 1

        # calculate element mass matrix
        me = (self.MASS_DENSITY * element_volume / 20) * M

        return me
    
    def __serialized_global_equation(self):
        """
        [K] {U} = {F}
        [K] {U} = {F_grav} + {F_buoy} + {F_contact} + {F_1} + {F_2} + {F_3} + ...
        """
        
        K = sp.lil_array((self.GLOBAL_DOF, self.GLOBAL_DOF))
        m = sp.lil_array((self.GLOBAL_DOF, self.GLOBAL_DOF))
        F = np.zeros((self.GLOBAL_DOF,), dtype=float)
        
        # consider element by element (contains 4 nodes)
        elements_progressbar = self.logger.track(
            sequence=self.domain.initial.cells_dict[pv.CellType.TETRA],
            description="Serialized Assembly",
            disable=not USE_PROGRESSBAR
        )
        for elemidx, nodeidx in enumerate(elements_progressbar):
            
            self.set_mechanical_properties(elemidx=elemidx)
            
            nodes_coord = self.domain.initial.points[nodeidx] # np.ndarray size (4, 3)
            dofidx = np.array([nodeidx * 3, nodeidx * 3 + 1, nodeidx * 3 + 2]).flatten('F') # np.ndarray size 12
            
            # calculate element matrices
            Ve = self.__tetrahedral_volume(nodes_coord=nodes_coord)
            Ke = sp.lil_array(self.__element_stiffness_matrix(nodes_coord=nodes_coord, element_volume=Ve))
            me = sp.lil_array(self.__element_mass_matrix(element_volume=Ve))
            
            # assemble global matrices
            for element_matrix_row, global_matrix_row in enumerate(dofidx):
                for element_matrix_col, global_matrix_col in enumerate(dofidx):
                    K[global_matrix_row, global_matrix_col] += Ke[element_matrix_row, element_matrix_col]
                    m[global_matrix_row, global_matrix_col] += me[element_matrix_row, element_matrix_col]
        
        return K, m, F
               
    def parallelized_assemble_element_matrices_task(self, nodeidx):
                
        nodes_coord = self.domain.initial.points[nodeidx] # np.ndarray size (4, 3)
        dofidx = np.array([nodeidx * 3, nodeidx * 3 + 1, nodeidx * 3 + 2]).flatten('F') # np.ndarray size 12
        
        # calculate element matrices
        Ve = self.__tetrahedral_volume(nodes_coord=nodes_coord)
        Ke = sp.lil_array(self.__element_stiffness_matrix(nodes_coord=nodes_coord, element_volume=Ve))
        me = sp.lil_array(self.__element_mass_matrix(element_volume=Ve))
        
        return dofidx, Ke, me
        
    def __parallelized_global_equation(self):
                
        K = sp.lil_array((self.GLOBAL_DOF, self.GLOBAL_DOF))
        m = sp.lil_array((self.GLOBAL_DOF, self.GLOBAL_DOF))
        F = np.zeros((self.GLOBAL_DOF,), dtype=float)
               
        with multiprocessing.Pool() as pool:
            
            elements_map = pool.map(
                self.parallelized_assemble_element_matrices_task,
                self.domain.initial.cells_dict[pv.CellType.TETRA]
            )
            
            elements_progressbar = self.logger.track(
                sequence=elements_map,
                description="Parallelized Assembly",
                disable=not USE_PROGRESSBAR,
            )
            for elemidx, (dofidx, Ke, me) in enumerate(elements_progressbar):
                # assemble global matrices
                dofidx_combination = itertools.product(enumerate(dofidx), enumerate(dofidx))
                for (element_matrix_row, global_matrix_row), (element_matrix_col, global_matrix_col) in dofidx_combination:
                    K[global_matrix_row, global_matrix_col] += Ke[element_matrix_row, element_matrix_col]
                    m[global_matrix_row, global_matrix_col] += me[element_matrix_row, element_matrix_col]
        
        return K, m, F
    
    def set_youngs_modulus(self, youngs_modulus: float):
        self.YOUNGS_MODULUS = youngs_modulus
        self.logger.loginfo(f"Young's modulus is set to {self.YOUNGS_MODULUS} Pa or g/(mm.s^2)")
        
    def set_poissons_ratio(self, poissons_ratio: float):
        self.POISSONS_RATIO = poissons_ratio
        self.logger.loginfo(f"Poisson's ratio is set to {self.POISSONS_RATIO}")
        
    def set_mass_density(self, mass_density: float):
        self.MASS_DENSITY = mass_density
        self.logger.loginfo(f"Mass density is set to {self.MASS_DENSITY} g/mm^3")
            
    def construct(self, parallelize: bool = True):
        if parallelize and not self.is_secondary_material_set:
            self.global_stiffness_matrix, self.global_mass_matrix, self.global_external_load_vector = self.__parallelized_global_equation()
        elif not parallelize or self.is_secondary_material_set:
            self.global_stiffness_matrix, self.global_mass_matrix, self.global_external_load_vector = self.__serialized_global_equation()
        
    @logger.logprocess("Set Secondary Material")    
    def set_secondary_material(self, youngs_modulus: float, poissons_ratio: float, mass_density: float, embedded_domain: EmbeddedDomain):

        # mechanical properties parameter adjustment
        
        # the concept is to find the elements that locate inside the surface mesh
        # and extract the element index.
        # then, create the condition to toggle the mechanical properties
        # during the stiffness matrix construction phase
                    
        # 1. instantiate the constants of primary and secondary material
        self.youngs_modulus_1 = self.YOUNGS_MODULUS
        self.poissons_ratio_1 = self.POISSONS_RATIO
        self.mass_density_1 = self.MASS_DENSITY
        self.youngs_modulus_2 = youngs_modulus
        self.poissons_ratio_2 = poissons_ratio
        self.mass_density_2 = mass_density
        self.embedded_domain = embedded_domain
        domain = self.domain.initial
        embedded_domain = embedded_domain.initial
        
        
        # 2. find the elements that locate inside the surface mesh
        #    using centroid of tetrahedral as an inclusion criteria
        
        #    boolean mask of tetrahedral elements
        domain_n_elem = domain.n_cells
        domain_tet_mask = np.full((domain_n_elem,), fill_value=False, dtype=bool)
        for i, c in enumerate(domain.cell):
            if c.type is pv.CellType.TETRA:
                domain_tet_mask[i] = True
                
        #    index of tetrahedral elements
        domain_tetidx = np.unique(np.where(domain_tet_mask)[0])
        
        #    centroid coord of tetrahedral elements
        domain_elemcentroid = domain.cell_centers()  # center of every cell types (point, line, triangular face, tetrahedral volume)
        domain_tetcentroid = pv.PolyData(domain_elemcentroid.points[domain_tetidx])
        
        #    associated nodes index in tetrahedral elements
        domain_tetnodeidx = domain.cells_dict[pv.CellType.TETRA]

        
        # 3. find points that are inside the surface mesh
        embedded_domain_surf = embedded_domain.extract_surface()
        selection = domain_tetcentroid.select_enclosed_points(embedded_domain_surf)
        selection_mask = selection["SelectedPoints"].view(bool)
        

        # 4. query the points that are inside the surface mesh based on the mask that we obtained
        
        #    element index of tetrahedral elements inside the surface
        domain_tetidx_in_surf = np.unique(np.where(selection_mask)[0])
        
        #    centroid coord of tetrahedral elements inside the surface
        domain_tetcentroid_in_surf = pv.PolyData(domain_tetcentroid.points[domain_tetidx_in_surf])
        
        #    node index of tetrahedral elements inside the surface
        domain_tetnodeidx_in_surf = np.unique(domain_tetnodeidx[domain_tetidx_in_surf])
        
        #    node coord of tetrahedral elements inside the surface
        domain_tetnode_in_surf = pv.PolyData(domain.points[domain_tetnodeidx_in_surf])
        
        
        # 5. assign the value into class variables
        self.domain_elemidx_mat2 = domain_tetidx_in_surf
        self.domain_elemidxmask_mat2 = selection_mask
        self.domain_elem_centroidcoord_mat2 = domain_tetcentroid_in_surf
        self.domain_elem_nodeidx_mat2 = domain_tetnodeidx_in_surf
        self.domain_elem_node_mat2 = domain_tetnode_in_surf

        # 6. toggle the flag
        self.is_secondary_material_set = True
            
    def set_mechanical_properties(self, elemidx: np.ndarray = None, elem_nodeidx: np.ndarray = None):
        
        if self.is_secondary_material_set:
            
            if elemidx is not None and elem_nodeidx is None:
                
                # primary material
                if elemidx not in self.domain_elemidx_mat2:
                    self.YOUNGS_MODULUS = self.youngs_modulus_1
                    self.POISSONS_RATIO = self.poissons_ratio_1
                    self.MASS_DENSITY = self.mass_density_1
                
                # secondary material
                elif elemidx in self.domain_elemidx_mat2:
                    self.YOUNGS_MODULUS = self.youngs_modulus_2
                    self.POISSONS_RATIO = self.poissons_ratio_2
                    self.MASS_DENSITY = self.mass_density_2
                                

class StructuralAnalysis:
    
    logger = TaskLogger("StructuralAnalysis")
    
    def __init__(self, domain: Domain):
        self.domain = domain
        self.domainsurf = self.domain.initial.extract_surface()
        self.domainsurf_nodeidx = self.domain.initial.surface_indices()
        self.GLOBAL_DOF = DOF_PER_NODE * self.domain.initial.n_points
        self.is_fixture_set = False
        self.is_buoyancy_set = False
        self.is_markers_set = False
        self.is_groundtruth_markers_set = False
        self.is_measured_domains_set = True
        self.is_contact_set = False
        self.tree_domain_node = cKDTree(self.domain.initial.points.astype(np.double))

    def __del__(self):
        pass
    
    def set_initial_condition(self):
        pass
    
    @logger.logprocess("Apply Boundary Condition")
    def set_boundary_condition(self, bc: Literal["fixture_2d_surface", "fixture_3d_volume", "fixture_3d_surface", "fixture_bbox_3d_surface", "gravity_3d_volume", "buoyancy_3d_surface"], **kwargs):
                
        def convert_valid_nodeidx_to_dofidx(domain_nodeidx_valid: np.ndarray):
            if domain_nodeidx_valid.size > 0:
                is_valid = True
                dofidx_idx = np.repeat(domain_nodeidx_valid, 3) * 3
                dofidx_axis = np.tile(np.array([0, 1, 2]), len(domain_nodeidx_valid))
                dofidx_valid = dofidx_idx + dofidx_axis
                nodes_valid = pv.PolyData(self.domain.initial.points[domain_nodeidx_valid, :])
            else:
                is_valid = False
                dofidx_valid = None
                nodes_valid = None
            return is_valid, dofidx_valid, nodes_valid
        
        bc_name = bc.replace("_", " ").title()
        self.logger.loginfo(f"Boundary Condition: [bold]{bc_name}[/bold].")

        # keyword arguments handling and instantiation
        if "referenced_plane" in kwargs:
            if kwargs["referenced_plane"] == "yz": axis = 0
            elif kwargs["referenced_plane"] == "xz": axis = 1
            elif kwargs["referenced_plane"] == "xy": axis = 2
        if "bounding_side" in kwargs:
            if kwargs["bounding_side"] == "lower": boundidx = axis * 2
            elif kwargs["bounding_side"] == "upper": boundidx = (axis * 2) + 1
        if "threshold" in kwargs:
            threshold = kwargs["threshold"]
        if "threshold_wrt_bounds" in kwargs:
            threshold_wrt_bounds = kwargs["threshold_wrt_bounds"]
            bounds = np.array(self.domain.initial.bounds)
            boundidx_min = axis * 2
            boundidx_max = boundidx_min + 1
            portion = threshold_wrt_bounds * abs(bounds[boundidx_max] - bounds[boundidx_min])
            threshold = bounds[boundidx] + portion
        if "bbox" in kwargs:
            bbox = kwargs["bbox"]
        if "fluid_level" in kwargs:
            fluid_level = kwargs["fluid_level"]
        if "direction" in kwargs:
            direction = kwargs["direction"]
        if "domain" in kwargs:
            if kwargs["domain"] == "initial":
                domain = self.domain.initial
            elif kwargs["domain"] == "deformed":
                domain = self.domain.deformed

        # find domain nodes indices that valid the boundary conditions
        if bc == "fixture_2d_surface":
            domain_nodeidx_fixed = np.array([], dtype=int)
            for nodeidx, node_coord in enumerate(self.domain.initial.points):
                if np.isclose(node_coord[axis], self.domain.initial.bounds[boundidx]):
                    domain_nodeidx_fixed = np.append(domain_nodeidx_fixed, nodeidx)
            self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(domain_nodeidx_fixed)
                    
        elif bc == "fixture_3d_volume":
            if direction == "lower":
                domain_nodeidx_fixed = np.where((self.domain.initial.points[:, axis] <= threshold))[0]
            elif direction == "upper":
                domain_nodeidx_fixed = np.where((self.domain.initial.points[:, axis] >= threshold))[0]
                    
            # remap fixed nodes indices from surface space to be global space
            if not self.is_fixture_set:
                self.domain_nodeidx_fixed = domain_nodeidx_fixed
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)
            elif self.is_fixture_set:
                new_domain_nodeidx_fixed = domain_nodeidx_fixed
                self.domain_nodeidx_fixed = np.append(self.domain_nodeidx_fixed, new_domain_nodeidx_fixed)
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)
        
        elif bc == "fixture_3d_surface":
            if direction == "lower":
                domainsurf_nodeidx_fixed = np.where((self.domainsurf.points[:, axis] <= threshold))[0]
            elif direction == "upper":
                domainsurf_nodeidx_fixed = np.where((self.domainsurf.points[:, axis] >= threshold))[0]
                
            # remap fixed nodes indices from surface space to be global space
            if not self.is_fixture_set:
                self.domain_nodeidx_fixed = self.domainsurf_nodeidx[domainsurf_nodeidx_fixed]
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)
            elif self.is_fixture_set:
                new_domain_nodeidx_fixed = self.domainsurf_nodeidx[domainsurf_nodeidx_fixed]
                self.domain_nodeidx_fixed = np.append(self.domain_nodeidx_fixed, new_domain_nodeidx_fixed)
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)

        elif bc == "fixture_bbox_3d_surface":

            if isinstance(bbox, np.ndarray):
                pass
            elif isinstance(bbox, str):
                if bbox == "auto_approx_brainstem":
                    FORAMEN_MAGNUM_WIDTH = 42
                    FORAMEN_MAGNUM_HALF_WIDTH = FORAMEN_MAGNUM_WIDTH / 2
                    bounds = self.domain.initial.bounds
                    x_brainstem = bounds[0] + (abs((bounds[1] - bounds[0])) / 3.5)
                    x_min = x_brainstem - FORAMEN_MAGNUM_HALF_WIDTH
                    x_max = x_brainstem + FORAMEN_MAGNUM_HALF_WIDTH
                    y_mid = (bounds[2] + bounds[3]) / 2
                    y_min = y_mid - FORAMEN_MAGNUM_HALF_WIDTH
                    y_max = y_mid + FORAMEN_MAGNUM_HALF_WIDTH
                    z_min = bounds[4]
                    z_max = z_min + 20
                    bbox = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

            domainsurf_elemidx_fixed = self.domainsurf.find_cells_within_bounds(bounds=bbox)
            domainsurf_elem_fixed = self.domainsurf.extract_cells(domainsurf_elemidx_fixed)
            domainsurf_node_fixed = domainsurf_elem_fixed.points
            tree_domainsurf_node = cKDTree(self.domainsurf.points.astype(np.double))
            dist, domainsurf_nodeidx_fixed = tree_domainsurf_node.query(domainsurf_node_fixed)

            # remap fixed nodes indices from surface space to be global space
            if not self.is_fixture_set:
                self.domain_nodeidx_fixed = self.domainsurf_nodeidx[domainsurf_nodeidx_fixed]
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)
            elif self.is_fixture_set:
                new_domain_nodeidx_fixed = self.domainsurf_nodeidx[domainsurf_nodeidx_fixed]
                self.domain_nodeidx_fixed = np.append(self.domain_nodeidx_fixed, new_domain_nodeidx_fixed)
                self.is_fixture_set, self.dofidx_fixed, self.node_fixed = convert_valid_nodeidx_to_dofidx(self.domain_nodeidx_fixed)

        elif bc == "gravity_3d_volume":
            # g = sp.lil_array((self.GLOBAL_DOF, 1))
            g = np.zeros((self.GLOBAL_DOF,), dtype=float)
            g[np.arange(2, self.GLOBAL_DOF, 3)] = -GRAVITATIONAL_ACCELERATION
            F_grav = self.material.global_mass_matrix @ g
            self.material.global_external_load_vector += F_grav
        
        elif bc == "buoyancy_3d_surface":

            # buoyancy force modeling:

            # find elemidx of surface cells from area
            A_global = domain.compute_cell_sizes().cell_data["Area"]  # face areas: (n_cells,)
            ei_surf = np.where(A_global != 0.0)[0]  # elemidx: (n_surf_cells,)
            
            # find the components of surface cells
            A_surf = A_global[ei_surf]  # face areas: (n_surf_cells,)
            n_surf = domain.extract_surface().triangulate().cell_normals  # normals: (n_surf_cells * 2, 3)
            c_surf = domain.extract_cells(ei_surf).cell_centers()  # face centroids: (n_surf_cells, 3)
            
            # find elemidx of submerged cells (submerged cells are the subset of surface cells)
            self.ei_sbmrg = np.where(c_surf.points[:, 2] <= fluid_level)[0]  # elemidx: (n_sbmrg_cells, 3)
            
            # find the components of submerged cells
            self.A_sbmrg = A_surf[self.ei_sbmrg]  # face areas: (n_sbmrg_cells,)
            self.n_sbmrg = n_surf[self.ei_sbmrg]  # normals: (n_sbmrg_cells, 3)
            self.ni_sbmrg = domain.cells_dict[pv.CellType.TRIANGLE][self.ei_sbmrg]  # nodeidx: (n_sbmrg_cells, 3)
            c_sbmrg = c_surf.points[self.ei_sbmrg]  # face centroids: (n_sbmrg_cells, 3)
            self.dh_sbmrg = np.abs(fluid_level - c_sbmrg[:, 2])  # submerged depths: (n_sbmrg_cells,)
            
            # (extra) filter the nodes that represented the embedded domain because it is also a surface
            if self.material.is_secondary_material_set:
                mask = ~np.isin(self.ni_sbmrg, self.material.domain_elem_nodeidx_mat2).any(axis=1)
                self.ni_sbmrg = self.ni_sbmrg[mask]

            # reinstate variables
            rho = constants(category="common", constant="water_mass_density") # g/mm^3
            g = abs(GRAVITATIONAL_ACCELERATION)
            dh = self.dh_sbmrg
            A = self.A_sbmrg
            n = self.n_sbmrg
            ni = self.ni_sbmrg
            
            # calculate local force vector acting on each face
            f = (rho * g * dh * A) / 3  # magnitude of force vector: (n_sbmrg_cells,)
            F_local = np.einsum("i,ij->ij", f, -n)  # force vector: (n_sbmrg_cells, 3)
            if self.material.is_secondary_material_set: F_local = F_local[mask, :]

            # find dofsidx to calculate global force vector
            F_local = np.repeat(F_local, 3, axis=0)
            F_global = np.zeros((domain.n_points * 3,), dtype=float)
            self.is_buoyancy_set, self.dofidx_submerged, self.node_submerged_initial = convert_valid_nodeidx_to_dofidx(ni.flatten())
            self.domain_face_nodeidx_submerged = ni

            # assemble global force vector
            np.add.at(F_global, self.dofidx_submerged, F_local.flatten())
            
            # add force vector to the global external load vector
            self.F_buoy = F_global
            self.material.global_external_load_vector += self.F_buoy

    def set_initial_markers(self, poly: pv.PolyData, mesh: pv.UnstructuredGrid):
        self.markers_initial_poly = poly
        self.markers_initial_mesh = mesh
        self.markers_initial_coord = self.markers_initial_poly.points
        
        # find global nodes indices of markers nodes using KDTree
        dist, self.markers_nodeidx = self.tree_domain_node.query(self.markers_initial_coord)
        if self.markers_nodeidx.size > 0: self.is_markers_set = True
    
    def set_initial_domain_orientation(self, goal_orientation: np.ndarray):
        self.domain.initial.rotate_x(goal_orientation[0], inplace=True)
        self.domain.initial.rotate_y(goal_orientation[1], inplace=True)
        self.domain.initial.rotate_z(goal_orientation[2], inplace=True)
        if self.is_fixture_set:
            self.node_fixed.rotate_x(goal_orientation[0], inplace=True)
            self.node_fixed.rotate_y(goal_orientation[1], inplace=True)
            self.node_fixed.rotate_z(goal_orientation[2], inplace=True)
        if self.is_markers_set:
            self.markers_initial_poly.rotate_x(goal_orientation[0], inplace=True)
            self.markers_initial_poly.rotate_y(goal_orientation[1], inplace=True)
            self.markers_initial_poly.rotate_z(goal_orientation[2], inplace=True)
            self.markers_initial_mesh.rotate_x(goal_orientation[0], inplace=True)
            self.markers_initial_mesh.rotate_y(goal_orientation[1], inplace=True)
            self.markers_initial_mesh.rotate_z(goal_orientation[2], inplace=True)
            self.markers_initial_coord = self.domain.initial.points[self.markers_nodeidx]
        if self.material.is_secondary_material_set:
            self.material.embedded_domain.initial.rotate_x(goal_orientation[0], inplace=True)
            self.material.embedded_domain.initial.rotate_y(goal_orientation[1], inplace=True)
            self.material.embedded_domain.initial.rotate_z(goal_orientation[2], inplace=True)
            self.material.embedded_domain.deformed.rotate_x(goal_orientation[0], inplace=True)
            self.material.embedded_domain.deformed.rotate_y(goal_orientation[1], inplace=True)
            self.material.embedded_domain.deformed.rotate_z(goal_orientation[2], inplace=True)
            self.lesion_initial_poly.rotate_x(goal_orientation[0], inplace=True)
            self.lesion_initial_poly.rotate_y(goal_orientation[1], inplace=True)
            self.lesion_initial_poly.rotate_z(goal_orientation[2], inplace=True)
            self.lesion_initial_elem_centroid_poly.rotate_x(goal_orientation[0], inplace=True)
            self.lesion_initial_elem_centroid_poly.rotate_y(goal_orientation[1], inplace=True)
            self.lesion_initial_elem_centroid_poly.rotate_z(goal_orientation[2], inplace=True)
        self.domainsurf = self.domain.initial.extract_surface()
        self.domainsurf_nodeidx = self.domain.initial.surface_indices()

    def set_initial_domain_position(self, goal_position: np.ndarray):
        self.domain.initial.translate(goal_position, inplace=True)
        if self.is_fixture_set:
            self.node_fixed.translate(goal_position, inplace=True)
        if self.is_markers_set:
            self.markers_initial_poly.translate(goal_position, inplace=True)     
        self.domainsurf = self.domain.initial.extract_surface()
        self.domainsurf_nodeidx = self.domain.initial.surface_indices()

    def set_groundtruth_deformed_markers(self, coord: np.ndarray, mesh: pv.UnstructuredGrid):
        self.markers_groundtruth_deformed_coord = coord
        self.markers_groundtruth_deformed_poly = pv.PolyData(self.markers_groundtruth_deformed_coord)
        self.markers_groundtruth_deformed_mesh = mesh
        self.is_groundtruth_markers_set = True
    
    def set_measured_domains(self, measured_initial: pv.UnstructuredGrid, measured_deformed: pv.UnstructuredGrid):
        self.measured_initial_domain = measured_initial
        self.measured_deformed_domain = measured_deformed
        self.is_measured_domains_set = True
    
    def set_deformed_domain_orientation(self, goal_orientation: np.ndarray):
        self.domain.deformed.rotate_x(goal_orientation[0], inplace=True)
        self.domain.deformed.rotate_y(goal_orientation[1], inplace=True)
        self.domain.deformed.rotate_z(goal_orientation[2], inplace=True)
         
    def set_deformed_domain_position(self, goal_position: np.ndarray):
        self.domain.deformed.translate(goal_position, inplace=True)
    
    def set_material_model(self, constitutive_law: LinearElasticModel):
        self.material = constitutive_law
        self.GLOBAL_DOF = self.material.GLOBAL_DOF
        self.global_displacement_vector = np.zeros((self.GLOBAL_DOF,), dtype=float)
        
        if self.material.is_secondary_material_set:
            self.lesion_nodeidx = np.unique(self.material.domain_elem_nodeidx_mat2)
            lesion_initial_coord = self.domain.initial.points[self.lesion_nodeidx, :]
            self.lesion_initial_poly = pv.PolyData(lesion_initial_coord)
            self.lesion_initial_elem_centroid_poly = pv.PolyData(self.material.domain_elem_centroidcoord_mat2)
            self.lesion_elemidx = self.query_lesion_elemidx(self.domain, self.material)
        
    def query_lesion_elemidx(self, domain: Domain, material: LinearElasticModel):
        tree = cKDTree(domain.initial.cell_centers().points)
        dist, lesion_elemidx = tree.query(material.domain_elem_centroidcoord_mat2.points.astype(np.double))
        return lesion_elemidx
    
    def construct_model(self, parallelize: bool = True):
        # assemble model
        self.material.construct(parallelize=parallelize)

        # create fixed dof mask and unfixed dof mask
        self.dof_fixed = np.zeros((self.GLOBAL_DOF,), dtype=bool)
        self.dof_fixed[self.dofidx_fixed] = True
        self.dof_unfixed = ~self.dof_fixed  # invert the logical values

        # stiffness matrix K, masking with unfixed nodes
        K_unfixed = sliced_matrix(self.material.global_stiffness_matrix, self.dof_unfixed)

        # setup solver
        self.solver = LUDecompositionSolver(device="cpu")
        # self.solver = ConjugateGradientSolver(preconditioner="jacobi", device="cpu")
        self.solver.setup(K_unfixed)
    
    @logger.logprocess("Solve")
    def solve(self):
        # to reduce the computation loads, the global matrix and vector are sliced
        # the entries that correlated with fixed DoFs will be sliced out,
        # so the entries that correlated with unfixed DoFs are remained for calculation
        F_unfixed = self.material.global_external_load_vector[self.dof_unfixed]
        self.__U_unfixed = self.solver.solve(F_unfixed)
        
    def postprocessing(self, update_on_deformed_domain: bool = False):
        
        # initialize new report
        self.report = Report()
        
        # construct global displacement vector and coord from unfixed dof displacement vector
        self.global_displacement_vector[self.dof_unfixed] = self.__U_unfixed
        self.global_displacement_coord = self.global_displacement_vector.reshape((-1, 3))
                        
        # update the deformation
        if not update_on_deformed_domain:
            self.domain.deformed = copy.deepcopy(self.domain.initial)
            self.domain.deformed.points = self.domain.initial.points + self.global_displacement_coord
            
        elif update_on_deformed_domain:
            self.domain.deformed.points = self.domain.deformed.points + self.global_displacement_coord
            
        total_displacement_coord = self.domain.deformed.points - self.domain.initial.points
        self.domain.deformed["Displacement"] = np.linalg.norm(total_displacement_coord, axis=1)
        self.report.set_bounds(self.domain.initial.bounds, self.domain.deformed.bounds)
        self.report.set_simulated_domains(self.domain.initial, self.domain.deformed, self.global_displacement_coord)
        
        if self.is_contact_set:
            self.report.set_contact_residual_penetration(self.max_penetration)
        
        if self.is_markers_set:
            self.markers_deformed_coord = self.domain.deformed.points[self.markers_nodeidx]
            self.markers_deformed_poly = pv.PolyData(self.markers_deformed_coord)
        
        if self.is_buoyancy_set:
            self.node_submerged_deformed = pv.PolyData(self.domain.deformed.points[np.unique(self.domain_face_nodeidx_submerged)])

        if self.material.is_secondary_material_set:
            domain_initial_volume = self.domain.get_volume(domain="initial", cell_indices=self.material.domain_elemidx_mat2)
            domain_deformed_volume = self.domain.get_volume(domain="deformed", cell_indices=self.material.domain_elemidx_mat2)
            self.report.set_lesion_volume(
                simulated_initial=domain_initial_volume,
                simulated_deformed=domain_deformed_volume,
            )
            lesion_deformed_coord = self.domain.deformed.points[self.lesion_nodeidx, :]
            self.lesion_deformed_poly = pv.PolyData(lesion_deformed_coord)
            self.report.set_lesion_coord(
                simulated_initial=self.lesion_initial_poly.points,
                simulated_deformed=self.lesion_deformed_poly.points,
                measured_initial=self.material.embedded_domain.initial.points,
                measured_deformed=self.material.embedded_domain.deformed.points,
            )
            if self.is_measured_domains_set:
                self.report.set_measured_domains(self.measured_initial_domain, self.measured_deformed_domain)

        report_table = self.report.get_report_table()
        # self.logger.loginfo(report_table)
        
    def get_info(self):
        return self.domain.initial, self.node_fixed, self.domain.deformed
    
    def set_contact(self, contact_mechanics: ContactMechanics | SelfContactMechanics):
        
        if type(contact_mechanics).__name__ == "ContactMechanics":
            self.logger.loginfo("Constraint Condition: [bold]Contact[/bold].")
            contact_name = "Contact"
        elif type(contact_mechanics).__name__ == "SelfContactMechanics":
            self.logger.loginfo("Constraint Condition: [bold]Self Contact[/bold].")
            contact_name = "Self Contact"
        else:
            self.logger.loginfo("Constraint Condition: [bold]Contact[/bold].")
            contact_name = "Contact"
        
        contact_mechanics.update_colliding_domain(self.domain.deformed)
        contact_mechanics.collision_detection()
        F_contact, n_iteration, self.max_penetration = contact_mechanics.contact_force()
        self.material.global_external_load_vector += F_contact
        self.is_contact_set = True

        report = (
            f"{contact_name} Iteration {n_iteration} "
            f"Max Penetration is {self.max_penetration:.2f} mm"
        )
        self.logger.loginfo(report)

    def visualize(self, mode: Literal["initial", "deformed"], show: bool = True):
        
        pv.set_plot_theme("default")
        self.plotter = pv.Plotter()
        
        def point_picking_callback(picked_mesh, picked_point_id):
            point = self.domain.deformed.node[picked_point_id]
            displacement = np.round(self.global_displacement_coord[picked_point_id], 4)
            label = [f"Node: {picked_point_id}\ndX: {displacement[0]} mm\ndY: {displacement[1]} mm\ndZ: {displacement[2]} mm"]
            self.plotter.add_point_labels(point, label, font_size=20)
                        
        def actor_initial_mesh_callback(state):
            self.actor_initial_mesh.SetVisibility(state)
            self.actor_initial_mesh_wireframe.SetVisibility(state)
                    
        def actor_deformed_mesh_callback(state):
            try:
                self.actor_deformed_mesh.SetVisibility(state)
                self.actor_deformed_mesh_wireframe.SetVisibility(state)
            except:
                pass
            
        def actor_node_fixed_callback(state):
            self.actor_node_fixed.SetVisibility(state)
                    
        def actor_node_submerged_callback(state):
            self.actor_node_submerged.SetVisibility(state)
 
        
        if mode == "initial":
            self.actor_initial_mesh = self.plotter.add_mesh(self.domain.initial, label="Initial Phantom", color="PaleVioletRed", lighting=True, opacity=0.5)
            self.actor_initial_mesh_wireframe = self.plotter.add_mesh(self.domain.initial, style="wireframe", color="Black", line_width=1, opacity=1)
            if self.is_fixture_set:
                self.actor_node_fixed = self.plotter.add_mesh(self.node_fixed, label="Fixed Nodes", color="Black", render_points_as_spheres=True, point_size=10)
            if self.is_buoyancy_set:
                self.actor_node_submerged = self.plotter.add_mesh(pv.PolyData(self.domain_face_centroidcoord_submerged), label="Submerged Nodes", color="Cyan", render_points_as_spheres=True, point_size=10)
                # self.actor_node_submerged = self.plotter.add_mesh(pv.PolyData(self.domain_face_nodescoord_submerged[1000]), label="Submerged Nodes", color="Green", render_points_as_spheres=True, point_size=10)
            if self.is_markers_set:
                self.actor_markers_initial_coord = self.plotter.add_mesh(self.markers_initial_poly, label="Initial Simulated Markers", color="LightCoral", render_points_as_spheres=True, point_size=10)
                self.actor_markers_initial_mesh = self.plotter.add_mesh(self.markers_initial_mesh, label="Initial Measured Markers (Mesh)", color="Cyan", opacity=0.2)
            if self.material.is_secondary_material_set:
                self.plotter.add_mesh(self.lesion_initial_elem_centroid_poly, label="Secondary Material Element Centroid", color="Green", render_points_as_spheres=True, point_size=10)
                self.plotter.add_mesh(self.lesion_initial_poly, label="Secondary Material Nodes", color="LightCoral", render_points_as_spheres=True, point_size=10)
                self.plotter.add_mesh(self.material.embedded_domain.initial, label="Initial Lesion", color="Green", lighting=True, opacity=0.8)
                self.plotter.add_mesh(self.material.embedded_domain.deformed, label="Shifted Lesion", color="Blue", lighting=True, opacity=0.8)

        elif mode == "deformed":
            self.actor_initial_mesh = self.plotter.add_mesh(self.domain.initial, label="Initial Phantom", color="LightGray", lighting=True, opacity=0.2, pickable=False)
            self.actor_initial_mesh_wireframe = self.plotter.add_mesh(self.domain.initial, style="wireframe", color="Black", line_width=1, opacity=1)
            
            if self.is_fixture_set:
                self.actor_node_fixed = self.plotter.add_mesh(self.node_fixed, label="Fixed Nodes", color="Black", render_points_as_spheres=True, point_size=10)
            if self.is_buoyancy_set:
                self.actor_node_submerged = self.plotter.add_mesh(self.node_submerged_deformed, label="Submerged Nodes", color="Cyan", render_points_as_spheres=True, point_size=10)
            if not self.is_markers_set and not self.material.is_secondary_material_set:
                self.actor_deformed_mesh = self.plotter.add_mesh(self.domain.deformed, label="Deformed Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)  
            if self.is_markers_set:
                self.actor_deformed_mesh = self.plotter.add_mesh(self.domain.deformed, label="Deformed Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=0.2)
                self.actor_markers_initial_coord = self.plotter.add_mesh(self.markers_initial_poly, label="Initial Simulated Markers", color="LightCoral", render_points_as_spheres=True, point_size=10)
                self.actor_markers_initial_mesh = self.plotter.add_mesh(self.markers_initial_mesh, label="Initial Measured Markers (Mesh)", color="Cyan", opacity=0.2)
                self.actor_markers_deformed_coord = self.plotter.add_mesh(self.markers_deformed_poly, label="Deformed Simulated Markers", color="Red", render_points_as_spheres=True, point_size=10)                
                for pi, pd in zip(self.markers_initial_coord, self.markers_deformed_coord):
                    self.plotter.add_mesh(pv.Arrow(pi, pd - pi, scale="auto"), label="Markers Displacement", color="Yellow", render_lines_as_tubes=True, line_width=5)
            if self.material.is_secondary_material_set:
                self.actor_deformed_mesh = self.plotter.add_mesh(self.domain.deformed, label="Deformed Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=0.2)
                self.actor_lesion_initial_coord = self.plotter.add_mesh(self.lesion_initial_poly, label="Initial Simulated Lesion", color="LightCoral", render_points_as_spheres=True, point_size=10)
                self.actor_lesion_deformed_coord = self.plotter.add_mesh(self.lesion_deformed_poly, label="Deformed Simulated Lesion", color="Red", render_points_as_spheres=True, point_size=10)                
                self.plotter.add_mesh(self.material.embedded_domain.deformed, label="Shifted Lesion", color="Blue", lighting=True, opacity=0.8)
                for pi, pd in zip(self.lesion_initial_poly.points, self.lesion_deformed_poly.points):
                    self.plotter.add_mesh(pv.Arrow(pi, pd - pi, scale="auto"), label="Lesion Displacement", color="Yellow", render_lines_as_tubes=True, line_width=5)

        self.plotter.enable_point_picking(callback=point_picking_callback, use_picker=True, font_size=10, render_points_as_spheres=True)
        self.plotter.add_checkbox_button_widget(callback=actor_initial_mesh_callback, value=True, color_on="LightGray", size=40, position=(10,10))
        self.plotter.add_checkbox_button_widget(callback=actor_deformed_mesh_callback, value=True, color_on="PaleVioletRed", size=40, position=(50,10))
        self.plotter.add_checkbox_button_widget(callback=actor_node_fixed_callback, value=True, color_on="Black", size=40, position=(90,10))
        self.plotter.add_checkbox_button_widget(callback=actor_node_submerged_callback, value=True, color_on="Cyan", size=40, position=(130,10))

        self.plotter.add_legend(bcolor=None)
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.9)
        
        if show:
            self.plotter.show()     

    def save_report(self, saved_directory):
        if saved_directory is None:
            self.logger.loginfo("Skip a report saving.")
        elif isinstance(saved_directory, str):
            self.report.save_report(saved_directory=saved_directory)

    def save_meshes(self, saved_directory):
        
        domain_initial = None
        domain_deformed = None
        embedded_domain_initial = None
        embedded_domain_deformed = None
        domain_nodeidx_submerged = None
        domain_nodeidx_fixed = None
        domain_elemidx_embedded_domain = None
        
        if saved_directory is None:
            self.logger.loginfo("Skip a report saving.")
            
        elif isinstance(saved_directory, str):
            os.makedirs(saved_directory, exist_ok=True)
            
            domain_initial = self.domain.initial
            domain_deformed = self.domain.deformed
            global_stiffness_matrix = self.material.global_stiffness_matrix
            global_external_load_vector = self.material.global_external_load_vector
            global_displacement_vector = self.global_displacement_vector
            if self.material.is_secondary_material_set:
                domain_elemidx_embedded_domain = self.lesion_elemidx
                embedded_domain_initial = domain_initial.extract_cells(self.lesion_elemidx)
                embedded_domain_deformed = domain_deformed.extract_cells(self.lesion_elemidx)
            if self.is_fixture_set:
                domain_nodeidx_fixed = self.domain_nodeidx_fixed
            if self.is_buoyancy_set:
                domain_nodeidx_submerged = self.domain_face_nodeidx_submerged
                
            data_to_save = {
                "domain_initial": domain_initial,
                "domain_deformed": domain_deformed,
                "embedded_domain_initial": embedded_domain_initial,
                "embedded_domain_deformed": embedded_domain_deformed,
                "domain_nodeidx_submerged": domain_nodeidx_submerged,
                "domain_nodeidx_fixed": domain_nodeidx_fixed,
                "domain_elemidx_embedded_domain": domain_elemidx_embedded_domain,
                "global_stiffness_matrix": global_stiffness_matrix,
                "global_external_load_vector": global_external_load_vector,
                "global_displacement_vector": global_displacement_vector,
            }
            
            for name, data in data_to_save.items():
                if data is not None:
                    if isinstance(data, pv.UnstructuredGrid):
                        saved_path = saved_directory + f"{name}.msh"
                        pv.save_meshio(saved_path, data)
                    elif isinstance(data, np.ndarray):
                        saved_path = saved_directory + f"{name}.npy"
                        np.save(saved_path, data)
                    elif isinstance(data, sp.lil_array):
                        data = data.tocsr()
                        saved_path = saved_directory + f"{name}.npz"
                        sp.save_npz(saved_path, data)
                    self.logger.loginfo(f"{saved_path} is successfully saved.")

