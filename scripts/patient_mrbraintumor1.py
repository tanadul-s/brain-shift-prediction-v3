from typing import Literal
import itertools

import numpy as np
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import sys
import path
root = path.Path(__file__).abspath().parent.parent
sys.path.append(root)
from src.utils.utils import parameters, constants, consoleconfigs, VIEW_VECTORS
from src.utils.logger import TaskLogger
from src.utils.markers_utils import MarkersRefframeFloatLesionHandler
from src.utils.result_utils import DeformationCurveResultHandler, MeshPostAnalyzer
from src.core.fem_structural_analysis import Domain, ScannedDomain, EmbeddedDomain, SegmentedDomain, LinearElasticModel, StructuralAnalysis
from src.core.contact_mechanics import ContactMechanics

# configs
LOG_LEVEL = consoleconfigs(config="logging_level")
USE_PROGRESSBAR = False


class MRBrainTumor1:
    
    logger = TaskLogger("MRBT1")
        
    def __init__(self) -> None:
        
        # parameters
        self.YOUNGS_MODULUS = constants(category="brain_mechanical_properties", constant="youngs_modulus")
        self.POISSONS_RATIO = constants(category="brain_mechanical_properties", constant="poissons_ratio")
        self.MASS_DENSITY = constants(category="brain_mechanical_properties", constant="mass_density")
        self.SOLVER = parameters(category="finite_element_method", parameter="solver")
        self.SOLVER = "sparse_cg"
        self.PENETRATION_TOLERANCE = 0.1
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 5
        self.fluid_portion = 0.75
        self.fixed_portion = 0.25

        # directory
        mesh_root_directory = r"data/patient_sample/mrbraintumor1/mesh/demo1_initial/"
        self.brain_directory = mesh_root_directory + r"brain.msh"
        self.lesion_directory = mesh_root_directory + r"lesion.msh"
        self.skull_directory = mesh_root_directory + r"skull.msh"
        self.result_root_directory = r"result/patient_sample/mrbraintumor1/demo1_test/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_orbiting_screenshot_root_directory = self.result_root_directory + r"orbiting_screenshots/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # orientation
        self.intrinsics_orientation = np.array([0, 0, -90])  # reorient components by a fixed rotation to standard orientation convention
        self.head_positionings = {
            "face_front": np.array([0, 0, 0]),
            "face_up": np.array([0, -90, 180]),
            "face_down": np.array([0, 90, 0]),
            "face_left": np.array([-90, 0, -90]),
            "face_right": np.array([90, 0, 90]),
            "face_up_oblique_left": np.array([-90, -45, -90]),
            "face_up_oblique_right": np.array([90, -45, 90]),
            "face_down_oblique_left": np.array([90, 45, 90]),            
            "face_down_oblique_right": np.array([-90, 45, -90]),
            "face_up_chin_up": np.array([0, -120, 180]),
        }  # reorient from standard orientation to get a desired head positioning
        self.initial_orientation = self.head_positionings["face_up_oblique_right"]
        self.register_to_sra_orientation = np.array([0, 0, 0])
        
    def small_deformation_problem(self, visualize: bool = True):
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")

        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)
        
        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)

        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: save report and meshes
        structural_analysis.save_report(self.saved_result_root_directory)

        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, skull
    
    def large_deformation_problem(self, visualize: bool = True):
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        
        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)
        
        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)
        
        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        
        # structural analysis: save report and meshes
        structural_analysis.save_report(self.saved_result_root_directory)

        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, skull
       
    def large_deformation_contact_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            try:
                colliding_points = structural_analysis.colliding_points
                collided_points = structural_analysis.collided_points
                plotter.add_mesh(mesh=colliding_points, color="magenta", point_size=10)
                plotter.add_mesh(mesh=collided_points, color="blue", point_size=10)
                for a, b in zip(colliding_points, collided_points):
                    ray = pv.Line(a, b)
                    plotter.add_mesh(ray, line_width=5, render_lines_as_tubes=True, color="yellow")
            except:
                pass
            __visualize_default_settings(plotter=plotter)
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()

        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)
        
        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)
 
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=brain.deformed, collided=skull.initial)
        prev_max_penetration = float("inf")
        patience_counter = 0
        for i in range(0, self.N_ITERATIONS_LIMIT):

            # set contact
            structural_analysis.set_contact(contact_mechanics=contact_mechanics)
            max_penetration = round(structural_analysis.max_penetration, 2)
            
            # solve the deformation
            structural_analysis.solve()
            structural_analysis.postprocessing()

            # check improvement of max_penetration
            if max_penetration >= prev_max_penetration:
                patience_counter += 1
                self.logger.loginfo(f"Penetration from contact is not improved. Attempt {patience_counter} out of {self.N_ITERATIONS_PATIENCE} iterations.")
                if patience_counter == self.N_ITERATIONS_PATIENCE:
                    self.logger.loginfo("Iteration patience limit is reached, contact formulation is stopped.")
                    break
            else:
                patience_counter = 0
            
            # update prev_max_penetration
            prev_max_penetration = max_penetration

            # check penetration tolerance
            if max_penetration <= self.PENETRATION_TOLERANCE:
                break
            elif max_penetration > self.PENETRATION_TOLERANCE:
                continue
  
        # structural analysis: save report and meshes
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)

        if visualize: 
            __visualize_deformed_after_contact()
            structural_analysis.visualize(mode="deformed")
        
        return structural_analysis, skull

    def large_deformation_buoyancy_problem(self, visualize: bool = True):
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        
        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)
        
        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)

        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: construct global equation
        structural_analysis.construct_model()
        
        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()            
       
        # structural analysis: apply buoyancy
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()
        fluid_level = brain_initial.bounds[4] + (abs(brain_initial.bounds[5] - brain_initial.bounds[4]) * self.fluid_portion)
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        
        # structural analysis: save report and meshes
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)

        if visualize: structural_analysis.visualize(mode="deformed")

        return structural_analysis, skull
    
    def large_deformation_contact_buoyancy_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            try:
                colliding_points = structural_analysis.colliding_points
                collided_points = structural_analysis.collided_points
                plotter.add_mesh(mesh=colliding_points, color="magenta", point_size=10)
                plotter.add_mesh(mesh=collided_points, color="blue", point_size=10)
                for a, b in zip(colliding_points, collided_points):
                    ray = pv.Line(a, b)
                    plotter.add_mesh(ray, line_width=5, render_lines_as_tubes=True, color="yellow")
            except:
                pass
            __visualize_default_settings(plotter=plotter)
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()

        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)
        
        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)

        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=brain.deformed, collided=skull.initial)
        prev_max_penetration = float("inf")
        patience_counter = 0
        for i in range(0, self.N_ITERATIONS_LIMIT):

            # set contact
            structural_analysis.set_contact(contact_mechanics=contact_mechanics)
            max_penetration = round(structural_analysis.max_penetration, 2)
            
            # solve the deformation
            structural_analysis.solve()
            structural_analysis.postprocessing()

            # check improvement of max_penetration
            if max_penetration >= prev_max_penetration:
                patience_counter += 1
                self.logger.loginfo(f"Penetration from contact is not improved. Attempt {patience_counter} out of {self.N_ITERATIONS_PATIENCE} iterations.")
                if patience_counter == self.N_ITERATIONS_PATIENCE:
                    self.logger.loginfo("Iteration patience limit is reached, contact formulation is stopped.")
                    break
            else:
                patience_counter = 0
            
            # update prev_max_penetration
            prev_max_penetration = max_penetration

            # check penetration tolerance
            if max_penetration <= self.PENETRATION_TOLERANCE:
                break
            elif max_penetration > self.PENETRATION_TOLERANCE:
                continue
  
        # structural analysis: apply buoyancy
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()
        fluid_level = brain_initial.bounds[4] + (abs(brain_initial.bounds[5] - brain_initial.bounds[4]) * self.fluid_portion)
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: save report and meshes        
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)

        if visualize: 
            __visualize_deformed_after_contact()
            structural_analysis.visualize(mode="deformed")
        
        return structural_analysis, skull

    def large_deformation_contact_buoyancy_contact_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=brain.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=brain.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(brain_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=skull.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            try:
                colliding_points = structural_analysis.colliding_points
                collided_points = structural_analysis.collided_points
                plotter.add_mesh(mesh=colliding_points, color="magenta", point_size=10)
                plotter.add_mesh(mesh=collided_points, color="blue", point_size=10)
                for a, b in zip(colliding_points, collided_points):
                    ray = pv.Line(a, b)
                    plotter.add_mesh(ray, line_width=5, render_lines_as_tubes=True, color="yellow")
            except:
                pass
            __visualize_default_settings(plotter=plotter)
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
                
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_measured_domains(brain.initial.copy(deep=True), brain.initial.copy(deep=True))
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()

        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="lesion_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="lesion_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="lesion_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)

        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)

        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)

        # if visualize: __visualize_initial()
        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=brain.deformed, collided=skull.initial)
        prev_max_penetration = float("inf")
        patience_counter = 0
        for i in range(0, self.N_ITERATIONS_LIMIT):

            # set contact
            structural_analysis.set_contact(contact_mechanics=contact_mechanics)
            max_penetration = round(structural_analysis.max_penetration, 2)
            
            # solve the deformation
            structural_analysis.solve()
            structural_analysis.postprocessing()

            # check improvement of max_penetration
            if max_penetration >= prev_max_penetration:
                patience_counter += 1
                self.logger.loginfo(f"Penetration from contact is not improved. Attempt {patience_counter} out of {self.N_ITERATIONS_PATIENCE} iterations.")
                if patience_counter == self.N_ITERATIONS_PATIENCE:
                    self.logger.loginfo("Iteration patience limit is reached, contact formulation is stopped.")
                    break
            else:
                patience_counter = 0
            
            # update prev_max_penetration
            prev_max_penetration = max_penetration

            # check penetration tolerance
            if max_penetration <= self.PENETRATION_TOLERANCE:
                break
            elif max_penetration > self.PENETRATION_TOLERANCE:
                continue
          
        # structural analysis: apply buoyancy
        brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()
        fluid_level = brain_initial.bounds[4] + (abs(brain_initial.bounds[5] - brain_initial.bounds[4]) * self.fluid_portion)
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=brain.deformed, collided=skull.initial)
        prev_max_penetration = float("inf")
        patience_counter = 0
        for i in range(0, self.N_ITERATIONS_LIMIT):

            # set contact
            structural_analysis.set_contact(contact_mechanics=contact_mechanics)
            max_penetration = round(structural_analysis.max_penetration, 2)
            
            # solve the deformation
            structural_analysis.solve()
            structural_analysis.postprocessing()

            # check improvement of max_penetration
            if max_penetration >= prev_max_penetration:
                patience_counter += 1
                self.logger.loginfo(f"Penetration from contact is not improved. Attempt {patience_counter} out of {self.N_ITERATIONS_PATIENCE} iterations.")
                if patience_counter == self.N_ITERATIONS_PATIENCE:
                    self.logger.loginfo("Iteration patience limit is reached, contact formulation is stopped.")
                    break
            else:
                patience_counter = 0
            
            # update prev_max_penetration
            prev_max_penetration = max_penetration

            # check penetration tolerance
            if max_penetration <= self.PENETRATION_TOLERANCE:
                break
            elif max_penetration > self.PENETRATION_TOLERANCE:
                continue
          
        # structural analysis: save report and meshes
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)

        if visualize: 
            __visualize_deformed_after_contact()
            structural_analysis.visualize(mode="deformed")
        
        return structural_analysis, skull

    def test(self, visualize: bool = True):
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=brain)
        structural_analysis.set_boundary_condition(bc="fixture_bbox_3d_surface", bbox="auto_approx_brainstem")
        
        # structural analysis: set constitutive model
        material = LinearElasticModel(domain=brain)
        material.set_youngs_modulus(self.YOUNGS_MODULUS)
        material.set_poissons_ratio(self.POISSONS_RATIO)
        material.set_mass_density(self.MASS_DENSITY)
        material.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        structural_analysis.set_material_model(constitutive_law=material)
        
        # structural analysis: set orientation
        structural_analysis.set_initial_domain_orientation(goal_orientation=self.initial_orientation)
        skull.set_orientation(goal_orientation=self.initial_orientation)

        # structural analysis: apply fixture after reorientation
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", bounding_side="lower", threshold_wrt_bounds=self.fixed_portion)

        if visualize: structural_analysis.visualize(mode="initial")
        # exit()

        bypass = True
        if bypass:
            import copy
            structural_analysis.domain.deformed = copy.deepcopy(structural_analysis.domain.initial)
            structural_analysis.domain.deformed["Displacement"] = np.zeros((structural_analysis.domain.initial.n_points,))
            lesion_deformed_coord = structural_analysis.domain.deformed.points[structural_analysis.lesion_nodeidx, :]
            structural_analysis.lesion_deformed_poly = pv.PolyData(lesion_deformed_coord)
            from scipy.sparse import lil_array
            structural_analysis.material.global_external_load_vector = lil_array((structural_analysis.GLOBAL_DOF, 1))
            # structural analysis: apply buoyancy
            brain_initial, brain_fixed_nodes, brain_deformed = structural_analysis.get_info()
            fluid_level = brain_initial.bounds[4] + (abs(brain_initial.bounds[5] - brain_initial.bounds[4]) * self.fluid_portion)
            structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=fluid_level, domain="deformed")
            structural_analysis.node_submerged_deformed = pv.PolyData(structural_analysis.domain.deformed.points[np.unique(structural_analysis.domain_face_nodeidx_submerged)])

        elif not bypass:
            # structural analysis: construct global equation
            structural_analysis.construct_model()

            # structural analysis: apply gravity
            structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
            structural_analysis.solve()
            structural_analysis.postprocessing()
            
            # structural analysis: save report and meshes
            structural_analysis.save_report(self.saved_result_root_directory)

        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, skull
   

class MRBrainTumor1Demonstration(MRBrainTumor1):

    logger = TaskLogger("MRBT1DEMO")

    def __init__(self):
        super().__init__()
     
    def demonstration_of_brain_shift_result_old(self, visualize: bool = True, save_screenshot: bool = False):
        
        def add_mesh_macros(configuration: Literal["initial", "deformed"], part: Literal["brain", "skull", "lesion", "lesion_nodes", "lesion_displacements", "fixed_nodes", "submerged_nodes"], opacity: float = 1.0, wireframe: bool = False):
            
            color_dict = {
                "initial": {
                    "skull": "LightGray",
                    "brain": "LightPink",
                    "lesion": "MediumTurquoise",
                    "fixed_nodes": "Black",
                    "submerged_nodes": "LightSkyBlue",
                },
                "deformed": {
                    "skull": "LightGray",
                    "brain": "Black",
                    "lesion": "OrangeRed",
                    "fixed_nodes": "Black",
                    "submerged_nodes": "LightSkyBlue",
                }
            }
            color = color_dict[configuration][part]

            if part == "brain":
                if configuration == "initial":
                    plotter.add_mesh(brain_initial, color=color, opacity=opacity, lighting=True)
                    if wireframe:
                        plotter.add_mesh(brain_initial, style="wireframe", color="Black", line_width=1, opacity=opacity)
                    legend_entries.append(["Initial Brain Phantom", color])
                elif configuration == "deformed":
                    # plotter.add_mesh(brain_deformed, show_edges=False, color=color, opacity=opacity, lighting=True)
                    plotter.add_mesh(brain_deformed, show_edges=False, scalars="Displacement", cmap="viridis", lighting=True, smooth_shading=True, opacity=opacity)
                    if wireframe:
                        plotter.add_mesh(brain_deformed, style="wireframe", color="Black", line_width=1, opacity=opacity)
                    legend_entries.append(["Deformed Brain Phantom", color])

            elif (configuration == "initial" or configuration == "deformed") and part == "fixed_nodes":
                plotter.add_mesh(pv.PolyData(node_fixed), color=color, point_size=10, render_points_as_spheres=True)
                legend_entries.append(["Fixed Nodes", color])

            elif (configuration == "initial" or configuration == "deformed") and part == "skull":
                plotter.add_mesh(mesh=skull.initial, color=color, lighting=True, opacity=opacity)
                if wireframe:
                    plotter.add_mesh(mesh=skull.initial, style="wireframe", color="Black", line_width=1, opacity=1)
                legend_entries.append(["Initial Skull Phantom", color])

            elif part == "lesion":
                if configuration == "initial":
                    plotter.add_mesh(mesh=lesion_initial_cells, color=color, opacity=opacity)
                    if wireframe:
                        plotter.add_mesh(mesh=lesion_initial_cells, style="wireframe", color="Black", line_width=1, opacity=1)
                    legend_entries.append(["Initial Lesion", color])
                elif configuration == "deformed":
                    plotter.add_mesh(mesh=lesion_deformed_cells, label="Deformed Lesion", color=color, opacity=opacity)
                    if wireframe:
                        plotter.add_mesh(mesh=lesion_deformed_cells, style="wireframe", color="Black", line_width=1, opacity=1)
                    legend_entries.append(["Deformed Lesion", color])

            elif configuration == "deformed" and part == "submerged_nodes":
                plotter.add_mesh(structural_analysis.node_submerged_deformed, label="Submerged Nodes", color=color, render_points_as_spheres=True, point_size=10)
                legend_entries.append(["Submerged Nodes", color])

        self.logger.loginfo("Demonstration of Brain Shift Result")
        
        # calculate the deformation
        # structural_analysis, skull = self.large_deformation_contact_buoyancy_contact_problem(visualize=False)
        structural_analysis, skull = self.test(visualize=False)
        brain_initial, node_fixed, brain_deformed = structural_analysis.get_info()
        lesion_elemidx = structural_analysis.lesion_elemidx
        lesion_initial_cells = brain_initial.extract_cells(lesion_elemidx)
        lesion_deformed_cells = brain_deformed.extract_cells(lesion_elemidx)

        # define constants for visualization and screenshots saving
        variants = list(range(0, 6))
        view_vectors = [
            VIEW_VECTORS["isometric_top"],
            VIEW_VECTORS["right"],
            VIEW_VECTORS["left"],
        ]
        view_configs = [0, 1, 2]
        wireframe_configs = [False, True]
        nodes_plotted_configs = [False, True]
        
        # visualization or screenshot saving
        if visualize or save_screenshot:
            visualization_configs = itertools.product(variants, view_configs, wireframe_configs, nodes_plotted_configs)
            for i_img, (level, i_view, is_wireframe, is_nodes_plotted) in enumerate(visualization_configs):

                if visualize: plotter = pv.Plotter(window_size=[1024, 1024])
                elif save_screenshot: plotter = pv.Plotter(window_size=[1024, 1024], off_screen=True)
                legend_entries = []
                view = view_vectors[i_view]

                if level == 0:
                    # initial all opaque
                    add_mesh_macros("initial", "skull", 1, is_wireframe)
                    add_mesh_macros("initial", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("initial", "fixed_nodes")

                elif level == 1:
                    # initial brain opaque
                    add_mesh_macros("initial", "skull", 0.1)
                    add_mesh_macros("initial", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("initial", "fixed_nodes")

                elif level == 2:
                    # initial lesion opaque
                    add_mesh_macros("initial", "skull", 0.1)
                    add_mesh_macros("initial", "brain", 0.2)
                    add_mesh_macros("initial", "lesion", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("initial", "fixed_nodes")

                elif level == 3:
                    # deformed all opaque
                    add_mesh_macros("initial", "skull", 1, is_wireframe)
                    add_mesh_macros("deformed", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("deformed", "submerged_nodes")
                        add_mesh_macros("deformed", "fixed_nodes")
                        
                elif level == 4:
                    # deformed brain opaque
                    add_mesh_macros("initial", "skull", 0.1)
                    add_mesh_macros("deformed", "brain", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("deformed", "submerged_nodes")
                        add_mesh_macros("deformed", "fixed_nodes")

                elif level == 5:
                    # deformed lesion opaque
                    add_mesh_macros("initial", "skull", 0.1)
                    add_mesh_macros("deformed", "brain", 0.2)
                    add_mesh_macros("deformed", "lesion", 1, is_wireframe)
                    if is_nodes_plotted:
                        add_mesh_macros("deformed", "submerged_nodes")
                        add_mesh_macros("deformed", "fixed_nodes")

                legend_entries, ind = np.unique(legend_entries, axis=0, return_index=True)
                legend_entries = legend_entries[np.argsort(ind)]
                plotter.add_axes()
                plotter.add_legend(legend_entries, bcolor=None)
                grid = plotter.show_grid()
                plotter.reset_camera()
                plotter.view_vector(view)
                plotter.camera.zoom(0.8)
                plotter.show(auto_close=False)
                if save_screenshot:
                    # make directory
                    os.makedirs(self.saved_screenshot_root_directory, exist_ok=True)
                    os.makedirs(self.saved_orbiting_screenshot_root_directory, exist_ok=True)

                    # file name
                    if i_view == 0: view_char = "i"
                    elif i_view == 1: view_char = "l"
                    elif i_view == 2: view_char = "r"
                    is_wireframe_char = "w" if is_wireframe else "x"
                    is_nodes_plotted_char = "n" if is_nodes_plotted else "x"

                    # save screenshot images (*.png)
                    screenshot_file_name = f"img{level + 1:02}_{view_char}_{is_wireframe_char}{is_nodes_plotted_char}"
                    screenshot_path = self.saved_screenshot_root_directory + f"{screenshot_file_name}.png"
                    plotter.screenshot(filename=screenshot_path)
                    self.logger.loginfo(f"The screenshot is saved at {screenshot_path}")

                    # save orbiting screenshot images (*.gif / *.mp4)
                    if i_view == len(view_configs) - 1:
                        orbiting_file_name = f"img{level + 1:02}_o_{is_wireframe_char}{is_nodes_plotted_char}"
                        orbiting_gif_path = self.saved_orbiting_screenshot_root_directory + f"{orbiting_file_name}.gif"
                        orbiting_mp4_path = self.saved_orbiting_screenshot_root_directory + f"{orbiting_file_name}.mp4"
                        path = plotter.generate_orbital_path(n_points=72, shift=0)
                        try:
                            plotter.remove_actor(grid)
                            plotter.remove_legend()
                            plotter.remove_scalar_bar()
                        except IndexError:
                            pass
                        plotter.open_gif(orbiting_gif_path)
                        plotter.orbit_on_path(path, write_frames=True)
                        plotter.open_movie(orbiting_mp4_path)
                        plotter.orbit_on_path(path, write_frames=True)
                        plotter.close()
                        self.logger.loginfo(f"The orbiting image is saved at {orbiting_gif_path}")
                        self.logger.loginfo(f"The orbiting movie is saved at {orbiting_mp4_path}")

    def demonstration_of_brain_shift_result(self, visualize: bool = True):
        
        self.logger.loginfo("Demonstration of Brain Shift Result")
        
        # calculate the deformation
        structural_analysis, skull = self.large_deformation_contact_buoyancy_contact_problem(visualize=False)
        # structural_analysis, skull = self.test(visualize=False)

    def save_screenshot(self):
        
        # brain
        brain = Domain(directory=self.brain_directory)
        brain.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # lesion
        lesion = EmbeddedDomain(directory_initial=self.lesion_directory, directory_deformed=self.lesion_directory)
        lesion.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        # skull
        skull = Domain(directory=self.skull_directory)
        skull.set_orientation(goal_orientation=self.intrinsics_orientation)
        
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            mesh=skull.initial,
            goal_orientation=self.initial_orientation
        )
        mpa.add_measured_meshes(
            domain_initial=brain.initial,
            domain_deformed=brain.initial,
            embedded_domain_initial=lesion.initial,
            embedded_domain_deformed=lesion.deformed,
            goal_orientation=self.initial_orientation
        )
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.result_analysis(self.saved_result_root_directory, register_to_sra=self.register_to_sra_orientation)
        mpa.visualize("save_screenshot", save_orbit=True, saved_screenshot_root_directory=self.saved_screenshot_root_directory, saved_graphic_root_directory=self.saved_graphic_root_directory, saved_orbiting_screenshot_root_directory=self.saved_orbiting_screenshot_root_directory)


class Demonstration1A(MRBrainTumor1Demonstration):

    logger = TaskLogger("DEMO1A")

    def __init__(self):
        super().__init__()

        # overwrite directory
        self.result_root_directory = r"result/patient_sample/mrbraintumor1/demo1a/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_orbiting_screenshot_root_directory = self.result_root_directory + r"orbiting_screenshots/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # overwrite configuration
        self.initial_orientation = self.head_positionings["face_up"]
        self.register_to_sra_orientation = np.array([0, 0, 0])
        self.fluid_portion = 0.75
        self.fixed_portion = 0.25
        

class Demonstration1B(MRBrainTumor1Demonstration):

    logger = TaskLogger("DEMO1B")

    def __init__(self):
        super().__init__()

        # overwrite directory
        self.result_root_directory = r"result/patient_sample/mrbraintumor1/demo1b/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_orbiting_screenshot_root_directory = self.result_root_directory + r"orbiting_screenshots/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # overwrite configuration
        self.initial_orientation = self.head_positionings["face_up_oblique_right"]
        self.register_to_sra_orientation = np.array([45, 0, 0])
        self.fluid_portion = 0.75
        self.fixed_portion = 0.25


class Demonstration1C(MRBrainTumor1Demonstration):

    logger = TaskLogger("DEMO1C")

    def __init__(self):
        super().__init__()

        # overwrite directory
        self.result_root_directory = r"result/patient_sample/mrbraintumor1/demo1c/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_orbiting_screenshot_root_directory = self.result_root_directory + r"orbiting_screenshots/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # overwrite configuration
        self.initial_orientation = self.head_positionings["face_right"]
        self.register_to_sra_orientation = np.array([90, 0, 0])
        self.fluid_portion = 0.75
        self.fixed_portion = 0.25


if __name__ == "__main__":
    # mrbraintumor1 = MRBrainTumor1()
    # mrbraintumor1.small_deformation_problem()
    # mrbraintumor1.large_deformation_problem()
    # mrbraintumor1.large_deformation_contact_problem()
    # mrbraintumor1.large_deformation_buoyancy_problem()
    # mrbraintumor1.large_deformation_contact_buoyancy_problem()
    # mrbraintumor1.large_deformation_contact_buoyancy_contact_problem() 
    # mrbraintumor1.test()

    demo1a = Demonstration1A()
    # demo1a.demonstration_of_brain_shift_result(visualize=False)
    demo1a.save_screenshot()
    del demo1a

    demo1b = Demonstration1B()
    # demo1b.demonstration_of_brain_shift_result(visualize=False)
    demo1b.save_screenshot()
    del demo1b

    demo1c = Demonstration1C()
    # demo1c.demonstration_of_brain_shift_result(visualize=False)
    demo1c.save_screenshot()
    del demo1c
    