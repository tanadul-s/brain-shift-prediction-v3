import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import pyvista as pv
from tqdm import tqdm

from src.utils.utils import constants, parameters, consoleconfigs
from src.utils.loggerold import CustomLogger, TimeElapsedLog
from src.utils.logger import TaskLogger

# configs
LOG_LEVEL = consoleconfigs(config="logging_level")
USE_PROGRESSBAR = False


class ContactMechanicsOld:
    
    logger = CustomLogger(name="CM", level=LOG_LEVEL)
    etime_logger = CustomLogger(name="CM", level=LOG_LEVEL)
    
    def __init__(self, colliding: pv.UnstructuredGrid, collided: pv.UnstructuredGrid, colliding_nodeidx_fixed: np.ndarray = None):
        # parameters
        self.CONTACT_STIFFNESS_CONSTANT = parameters(category="contact_mechanics", parameter="contact_stiffness_factor")
        
        # meshes
        self.colliding = colliding  # brain
        self.collided = collided  # skull

        # fixed nodes of colliding domain
        self.colliding_surface_nodeidx = self.colliding.surface_indices()
        if isinstance(colliding_nodeidx_fixed, np.ndarray):
            self.colliding_nodeidx_fixed = colliding_nodeidx_fixed
            self.colliding_surface_nodeidx_unfixed = np.setdiff1d(self.colliding_surface_nodeidx, self.colliding_nodeidx_fixed)
        else:
            self.colliding_surface_nodeidx_unfixed = self.colliding_surface_nodeidx

        # counter
        self.iteration_counter = 0

    def collision_detection(self):
        
        obj1 = self.colliding.extract_surface()  # brain
        obj2 = self.collided.extract_surface()  # skull
        
        p1s = obj1.points[self.colliding_surface_nodeidx_unfixed]
        n1s = obj1.point_normals[self.colliding_surface_nodeidx_unfixed]
        n2s = obj2.cell_normals
        
        magnitude = 10000
        self.colliding_points = np.empty(shape=(0, 3))
        self.collided_points = np.empty(shape=(0, 3))
        self.collided_cells = np.array([], dtype=int)
        self.iteration_counter += 1
        
        self.logger.info("Perform a collision detection...")
        collision_detection_progressbar = tqdm(
            zip(p1s, n1s),
            total=len(p1s),
            desc=f"Collision Detection {self.iteration_counter}",
            ncols=80,
            disable=not USE_PROGRESSBAR
        )
        for p1, n1 in collision_detection_progressbar:
            
            p2_candidate, c2_candidate = obj2.ray_trace(origin=p1, end_point=p1 + (-n1 * magnitude), first_point=True)

            if np.size(p2_candidate):
                p2 = p2_candidate
                c2 = c2_candidate[0]
                n2 = n2s[c2]
                
                if np.dot(n1, n2) > 0:
                    continue
                
                p3_candidate, c3_candidate = obj2.ray_trace(origin=p2, end_point=p2 + (n1 * magnitude), first_point=False)
                if len(p3_candidate) > 1 and np.array_equal(p3_candidate[0], p2):
                    p3 = p3_candidate[1]  # because p3_candidate[0] must be p2
                    if np.linalg.norm(p2 - p1) <= np.linalg.norm(p2 - p3):
                        self.colliding_points = np.append(self.colliding_points, [p1], axis=0)
                        self.collided_points = np.append(self.collided_points, [p2], axis=0)
                        self.collided_cells = np.append(self.collided_cells, c2)
            
    def contact_force(self):
    
        colliding_nodeidx = np.array([], dtype=int)
        for p in self.colliding_points:
            nodeidx = self.colliding.find_closest_point(p)
            colliding_nodeidx = np.append(colliding_nodeidx, nodeidx)
        
        # normal_vectors = collided.extract_surface().cell_normals[collided_cells]
        penetration_vectors = self.collided_points - self.colliding_points
        penetration_distances = np.linalg.norm(penetration_vectors, axis=1)[:, np.newaxis]
        penetration_directions = penetration_vectors / penetration_distances  # unit vectors
        
        # construct contact forces vector
        F = np.zeros((self.colliding.n_points * 3, 1))

        if len(penetration_distances) > 0:
            max_penetration = np.amax(penetration_distances)
            contact_stiffness_constant = self.__contact_stiffness_constant(method="youngs_modulus")
            contact_forces = contact_stiffness_constant * penetration_vectors
            for i, f in zip(colliding_nodeidx, contact_forces):
                F[i * 3] = f[0]
                F[i * 3 + 1] = f[1]
                F[i * 3 + 2] = f[2]

        elif len(penetration_distances) == 0:
            max_penetration = 0

        return sp.lil_array(F), self.iteration_counter, max_penetration
    
    def __contact_stiffness_constant(self, method: str = "constant"):
        if method == "constant":
            return self.CONTACT_STIFFNESS_CONSTANT
        
        if method == "youngs_modulus":
            E = constants(category="brain_mechanical_properties", constant="youngs_modulus")
            gain = 1.0
            return gain * E

    def update_colliding_domain(self, colliding: pv.UnstructuredGrid):
        self.colliding = colliding  # brain

class SelfContactMechanics(ContactMechanicsOld):
    
    logger = CustomLogger(name="SCM", level=LOG_LEVEL)
    etime_logger = CustomLogger(name="SCM", level=LOG_LEVEL)
    
    def __init__(self, domain: pv.UnstructuredGrid):
        
        # parameters
        self.CONTACT_STIFFNESS_CONSTANT = parameters(category="contact_mechanics", parameter="contact_stiffness_factor")
        
        # meshes
        self.domain = domain
        self.colliding = domain
        self.collided = domain
        
        # counter
        self.iteration_counter = 0
        
    def collision_detection(self):
        
        obj1 = self.colliding.extract_surface().triangulate().compute_normals()
        obj2 = self.collided.extract_surface().triangulate().compute_normals()
        
        p1s = obj1.points
        n1s = obj1.point_normals
        n2s = obj2.cell_normals
        
        magnitude = 10000
        self.colliding_points = np.empty(shape=(0, 3))
        self.collided_points = np.empty(shape=(0, 3))
        self.collided_cells = np.array([], dtype=int)
        self.iteration_counter += 1
                
        self.logger.info("Perform a self collision detection...")
        collision_detection_progressbar = tqdm(
            zip(p1s, n1s),
            total=len(p1s),
            desc=f"Collision Detection {self.iteration_counter}",
            ncols=80,
            disable=True
        )
        
        for p1, n1 in collision_detection_progressbar:
           
            p2_candidates, intersected_rays, c2_candidates = obj1.multi_ray_trace([p1], [-n1], first_point=False)

            if len(p2_candidates) == 3:
                
                # has 3 intersected points because the first point is p1 itself
                # the second and third points are actual intersected points
                
                # sort the p2_candidates based on Euclidean distances between itself and p1
                distances = np.linalg.norm(p1 - p2_candidates, axis=1)
                sorted_indices = np.argsort(distances)
                p2_candidates = p2_candidates[sorted_indices]  # sorted p2_candidates
                c2_candidates = c2_candidates[sorted_indices]  # sorted c2_candidates
                p2 = p2_candidates[1]  # 1st intersected points of inward traced ray (true p2)
                p2_val = p2_candidates[2]  # 2nd intersected points of inward traced ray (validating p2)
                c2 = c2_candidates[1]
                n2 = n2s[c2]
                
                if np.dot(n1, n2) > 0:
                    continue
                
                self.colliding_points = np.append(self.colliding_points, [p1], axis=0)
                self.collided_points = np.append(self.collided_points, [p2], axis=0)
                self.collided_cells = np.append(self.collided_cells, c2)
                

class ContactMechanics:
    
    logger = TaskLogger("ContactMechanics")
    
    def __init__(self, colliding: pv.UnstructuredGrid, collided: pv.UnstructuredGrid, colliding_nodeidx_fixed: np.ndarray = None):
        # parameters
        self.CONTACT_STIFFNESS_CONSTANT = parameters(category="contact_mechanics", parameter="contact_stiffness_factor")
        
        # meshes
        self.colliding = colliding  # brain
        self.collided = collided  # skull
        self.colliding_tree = cKDTree(self.colliding.points)

        # counter
        self.iteration_counter = 0

    @logger.logprocess("Collision Detection")
    def collision_detection(self):
        
        obj1 = self.colliding.extract_surface().triangulate()  # deformable body
        obj2 = self.collided.extract_surface().triangulate()  # rigid body

        p1s = obj1.points
        n1s = obj1.point_normals
        n2s = obj2.cell_normals
       
        # step 1: find candidate collision pairs
        p2_cands, p1_cands_idx, c2_cands_idx = obj2.multi_ray_trace(origins=p1s, directions=-n1s, first_point=True)
        
        # step 2: find the true collision pairs from the candidates
        if np.size(p2_cands) > 0:
            
            # query points and normals, scoped to the candidates
            p2s = p2_cands
            n2s = n2s[c2_cands_idx]
            n1s = n1s[p1_cands_idx]
            
            # filter out the pairs that do not create acute angles (>= 90 degrees; dot product <= 0)
            dot_prods = np.einsum("ij,ij->i", n1s, n2s)  # dot products of n1 and n2 along the rows
            cand_pairs_idx = (dot_prods <= 0)
            
            # query points and normals, scoped to the candidates
            p2s = p2s[cand_pairs_idx]
            p1s = p1s[p1_cands_idx][cand_pairs_idx]
            n1s = n1s[cand_pairs_idx]
           
            # find candidate validating points
            p3s, _, _ = obj2.multi_ray_trace(origins=p2s, directions=n1s, first_point=True)
            
            # filter out false candidate pairs
            d21s = np.linalg.norm(p2s - p1s, axis=1)
            d23s = np.linalg.norm(p2s - p3s, axis=1)
            true_pairs_idx = (d21s <= d23s)

            # query true collision pairs
            self._colliding_points = p1s[true_pairs_idx]
            self._collided_points = p2s[true_pairs_idx]
        
        else:
            self._colliding_points = np.empty((0, 3), dtype=float)
            self._collided_points = np.empty((0, 3), dtype=float)
        
        self.iteration_counter += 1

    @logger.logprocess("Contact Load Modeling")
    def contact_force(self):

        self._n_points = len(self._colliding_points)  # number of collision pairs
        self._n_dofs = self._n_points * 3  # number of degrees of freedom
        self._d = self._collided_points - self._colliding_points  # penetration vectors
        self._k = self.__contact_stiffness_constant("constant")

        Flocal = np.zeros((self._n_dofs,))
        F = np.zeros((self.colliding.n_points * 3,), dtype=float)
        if self._n_points > 0:
            max_penetration = np.amax(np.linalg.norm(self._d, axis=1))
            _, colliding_nodeidx = self.colliding_tree.query(self._colliding_points)
            colliding_dofsidx = np.array([colliding_nodeidx * 3, colliding_nodeidx * 3 + 1, colliding_nodeidx * 3 + 2]).flatten("F")
            Flocal = self._k * self._d.flatten()
            F[colliding_dofsidx] = Flocal
        elif self._n_points == 0:
            max_penetration = 0

        return F, self.iteration_counter, max_penetration
    
    def __contact_stiffness_constant(self, method: str = "constant"):
        if method == "constant":
            return self.CONTACT_STIFFNESS_CONSTANT
        
        if method == "youngs_modulus":
            E = constants(category="brain_mechanical_properties", constant="youngs_modulus")
            gain = 1.0
            return gain * E

    def update_colliding_domain(self, colliding: pv.UnstructuredGrid):
        self.colliding = colliding  # brain

