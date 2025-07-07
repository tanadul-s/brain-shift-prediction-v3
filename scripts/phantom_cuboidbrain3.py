from typing import Literal

import numpy as np
import pyvista as pv

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


class CuboidBrain3:
    
    logger = TaskLogger("CB3")
    
    def __init__(self) -> None:
        
        # parameters
        self.PENETRATION_TOLERANCE = parameters(category="contact_mechanics", parameter="penetration_tolerance")
        self.N_ITERATIONS_LIMIT = parameters(category="contact_mechanics", parameter="n_iterations_limit")
        self.N_ITERATIONS_PATIENCE = parameters(category="contact_mechanics", parameter="n_iterations_patience")
        self.fluid_level = 0

        self.phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_with_lesion.msh"
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4_test/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"
       
    def segmented_meshes_registration(self, expt: Literal["a", "b", "c"], visualize: bool = False):
                
        np.set_printoptions(suppress=True)
        
        # initial refframe in CAD
        initial_refframe_o = np.array([140.0, 0.0, 0.0])
        initial_refframe_dict = {
            "o": initial_refframe_o,
            "x": initial_refframe_o + np.array([5.0, 0.0, 0.0]),
            "y": initial_refframe_o + np.array([0.0, 10.0, 0.0]),
            "z": initial_refframe_o + np.array([0.0, 0.0, 15.0]),
        }
        initial_refframe_array = np.stack(list(initial_refframe_dict.values()))  # xyz
        
        # deformed refframe in CAD
        deformed_refframe_o = np.array([0.0, 0.0, 120.0])
        deformed_refframe_dict = {
            "o": deformed_refframe_o,
            "x": deformed_refframe_o + np.array([5.0, 0.0, 0.0]),
            "y": deformed_refframe_o + np.array([0.0, 10.0, 0.0]),
            "z": deformed_refframe_o + np.array([0.0, 0.0, 15.0]),
        }
        deformed_refframe_array = np.stack(list(deformed_refframe_dict.values()))  # xyz
        
        # registration
        initial_markers_root_path = r"data/phantom/cuboidbrain3/scanned/expt4_initial_scan_processing/"
        initial_mesh_root_path = r"data/phantom/cuboidbrain3/mesh/measured/expt4_initial/"
        deformed_markers_root_path = r"data/phantom/cuboidbrain3/scanned/expt4_deformed_scan_processing/"
        deformed_mesh_root_path = rf"data/phantom/cuboidbrain3/mesh/measured/expt4{expt}/"
        handler = MarkersRefframeFloatLesionHandler(
            initial_markers_root_path=initial_markers_root_path,
            initial_mesh_root_path=initial_mesh_root_path,
            initial_refframe_dict=initial_refframe_dict,
            initial_markers_file_name="markers_initial",
            deformed_markers_root_path=deformed_markers_root_path,
            deformed_mesh_root_path=deformed_mesh_root_path,
            deformed_refframe_dict=deformed_refframe_dict,
            deformed_markers_file_name=f"markers_deformed_{expt}",
            segmented_brain=SegmentedDomain(directory_initial=initial_mesh_root_path + r"segmented_initial_silicone.msh", directory_deformed=deformed_mesh_root_path + r"segmented_deformed_silicone.msh"),
            segmented_markers=SegmentedDomain(directory_initial=initial_mesh_root_path + r"segmented_initial_markers.msh", directory_deformed=deformed_mesh_root_path + r"segmented_deformed_markers.msh"),
            segmented_lesion=SegmentedDomain(directory_initial=initial_mesh_root_path + r"segmented_initial_lesion.msh", directory_deformed=deformed_mesh_root_path + r"segmented_deformed_lesion.msh"),
        )
        handler.registration(domain="initial")
        handler.registration(domain="deformed")
        
        # global transformation
        
        # to transform my correct axes to gmsh axes; to embed point and generate mesh
        # (z points forward; y points left; x points upward)
        rotate_system_to_gmsh = np.array([-90, -90, 0])
        # remark: no domain orientation is set before applying fixture
        
        # to align initial to deformed; to register the initial and deformed mesh together
        # (x points forward; y points left; z points upward)
        align_initial_to_deformed = np.array([-90, -90, -90])
        # remark: the domain orientation is set (rotate z by -90deg) after fixture is set

        handler.set_global_transformation(
            domain="initial",
            rotate=rotate_system_to_gmsh,
            rotate_about_point=np.array([0, 0, 0])
        ) 
        
        handler.set_global_transformation(
            domain="deformed",
            rotate=np.array([0, 0, 90]),
            rotate_about_point=np.array([0, 0, 0])
        ) 

        # instantiate registered mesh
        self.registered_initial_markers = handler.segmented_markers.initial
        self.registered_initial_brain = handler.segmented_brain.initial
        self.registered_initial_lesion = handler.segmented_lesion.initial
        self.registered_deformed_markers = handler.segmented_markers.deformed
        self.registered_deformed_brain = handler.segmented_brain.deformed
        self.registered_deformed_lesion = handler.segmented_lesion.deformed
        
        # visualize
        if visualize:
            plotter = pv.Plotter()
            
            plotter.add_mesh(initial_refframe_array, label="initial true refframe", color="red", render_points_as_spheres=True, point_size=10)
            plotter.add_mesh(handler.segmented_markers.initial, label="initial markers", color="pink", opacity=0.5)
            plotter.add_mesh(handler.segmented_brain.initial, label="initial brain", color="pink", opacity=0.5)
            plotter.add_mesh(handler.segmented_lesion.initial, label="initial lesion", color="pink", opacity=0.5)

            plotter.add_mesh(deformed_refframe_array, label="deformed true refframe", color="blue", render_points_as_spheres=True, point_size=10)
            plotter.add_mesh(handler.segmented_markers.deformed, label="deformed markers", color="cyan", opacity=0.5)
            plotter.add_mesh(handler.segmented_brain.deformed, label="deformed brain", color="cyan", opacity=0.5)
            plotter.add_mesh(handler.segmented_lesion.deformed, label="deformed lesion", color="cyan", opacity=0.5)

            plotter.add_legend()
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()  

    def small_deformation_problem(self, visualize: bool = True):
        
        phantom = Domain(directory=self.phantom_directory)
        
        structural_analysis = StructuralAnalysis(domain=phantom)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xy", direction="lower", threshold=0.5)
        if visualize: structural_analysis.visualize(mode="initial")

        model = LinearElasticModel(domain=phantom)
                
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.construct_model()
        structural_analysis.solve()
        structural_analysis.postprocessing()
                
        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis
    
    def large_deformation_problem(self, visualize: bool = True):
               
        # phantom
        phantom = Domain(grid=self.registered_initial_brain)
        phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))

        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
            
        # constitutive model
        model = LinearElasticModel(domain=phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        
        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: contruct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: structural_analysis.visualize(mode="deformed")
                    
        return structural_analysis, collided_phantom
     
    def large_deformation_contact_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
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
        
        # colliding phantom
        colliding_phantom = Domain(grid=self.registered_initial_brain)
        colliding_phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))
        
        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=colliding_phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
        colliding_phantom_initial, colliding_phantom_fixed_nodes, colliding_phantom_deformed = structural_analysis.get_info()
                
        # constitutive model
        model = LinearElasticModel(domain=colliding_phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )

        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=colliding_phantom.deformed, collided=collided_phantom.initial)
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
  
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)

        if visualize: 
            __visualize_deformed_after_contact()
            structural_analysis.visualize(mode="deformed")
        
        return structural_analysis, collided_phantom

    def large_deformation_buoyancy_problem(self, visualize: bool = True):
               
        # phantom
        phantom = Domain(grid=self.registered_initial_brain)
        phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))

        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # structural analysis: apply fixture
        structural_analysis = StructuralAnalysis(domain=phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
            
        # constitutive model
        model = LinearElasticModel(domain=phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )

        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: structural_analysis.visualize(mode="initial")

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: apply buoyancy
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=self.fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: structural_analysis.visualize(mode="deformed")

        return structural_analysis, collided_phantom
    
    def large_deformation_contact_buoyancy_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
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
        
        # colliding phantom
        colliding_phantom = Domain(grid=self.registered_initial_brain)
        colliding_phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))
        
        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # structural analysis: set fixture
        structural_analysis = StructuralAnalysis(domain=colliding_phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
        colliding_phantom_initial, colliding_phantom_fixed_nodes, colliding_phantom_deformed = structural_analysis.get_info()
        
        # constitutive model
        model = LinearElasticModel(domain=colliding_phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        
        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=colliding_phantom.deformed, collided=collided_phantom.initial)
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
          
        # if visualize: __visualize_deformed_after_contact()

        # structural analysis: apply buoyancy
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=self.fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: __visualize_deformed()
        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, collided_phantom

    def large_deformation_buoyancy_contact_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
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
        
        # colliding phantom
        colliding_phantom = Domain(grid=self.registered_initial_brain)
        colliding_phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))
        
        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # structural analysis: set fixture
        structural_analysis = StructuralAnalysis(domain=colliding_phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
        colliding_phantom_initial, colliding_phantom_fixed_nodes, colliding_phantom_deformed = structural_analysis.get_info()
        
        # constitutive model
        model = LinearElasticModel(domain=colliding_phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        
        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply buoyancy
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=self.fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=colliding_phantom.deformed, collided=collided_phantom.initial)
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
            
        # if visualize: __visualize_deformed_after_contact()

        # structural analysis: save report and meshes
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: __visualize_deformed()
        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, collided_phantom    
    
    def large_deformation_contact_buoyancy_contact_problem(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
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
        
        # colliding phantom
        colliding_phantom = Domain(grid=self.registered_initial_brain)
        colliding_phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))
        
        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # structural analysis: set fixture
        structural_analysis = StructuralAnalysis(domain=colliding_phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
        colliding_phantom_initial, colliding_phantom_fixed_nodes, colliding_phantom_deformed = structural_analysis.get_info()
        
        # constitutive model
        model = LinearElasticModel(domain=colliding_phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        
        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()
        # if visualize: __visualize_deformed()
        
        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=colliding_phantom.deformed, collided=collided_phantom.initial)
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
            
        # if visualize: __visualize_deformed_after_contact()

        # structural analysis: apply buoyancy
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=self.fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: apply contact
        contact_mechanics = ContactMechanics(colliding=colliding_phantom.deformed, collided=collided_phantom.initial)
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
            
        # if visualize: __visualize_deformed_after_contact()

        # structural analysis: save report and meshes
        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: __visualize_deformed()
        if visualize: structural_analysis.visualize(mode="deformed")
            
        return structural_analysis, collided_phantom    
   
    def test(self, visualize: bool = True):
        
        def __visualize_default_settings(plotter: pv.Plotter):
            plotter.add_legend(bcolor=None)
            plotter.add_axes()
            plotter.show_grid()
            plotter.reset_camera()
            plotter.show()    
        
        def __visualize_initial():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=1)
            __visualize_default_settings(plotter=plotter)
            
        def __visualize_deformed():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
            __visualize_default_settings(plotter=plotter)
        
        def __visualize_deformed_after_contact():
            plotter = pv.Plotter()
            plotter.add_mesh(mesh=colliding_phantom.initial, label="Initial Colliding Phantom", color="LightPink", show_edges=True, opacity=0.2)
            # plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom ", color="PaleVioletRed", show_edges=True, opacity=0.8)
            plotter.add_mesh(mesh=colliding_phantom.deformed, label="Deformed Colliding Phantom", scalars="Displacement", cmap="viridis", lighting=True, opacity=1)
            plotter.add_mesh(mesh=pv.PolyData(colliding_phantom_fixed_nodes), label="Fixed Nodes", color="Black", point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(mesh=collided_phantom.initial, label="Collided Phantom", color="LightGrey", show_edges=True, opacity=0.2)
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
        
        # colliding phantom
        colliding_phantom = Domain(grid=self.registered_initial_brain)
        colliding_phantom.set_orientation(goal_orientation=np.array([0, 0, 0]))
        
        # collided phantom
        collided_phantom_directory = r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh"
        collided_phantom = Domain(directory=collided_phantom_directory)
        collided_phantom.set_orientation(goal_orientation=np.array([90, 0, 0]))
        collided_phantom.set_position(goal_position=np.array([0, 0, 0]))
        
        # lesion
        lesion = EmbeddedDomain(grid_initial=self.registered_initial_lesion, grid_deformed=self.registered_deformed_lesion)
        lesion.set_orientation(goal_orientation=np.array([0, 0, 0]))

        # structural analysis: set fixture
        structural_analysis = StructuralAnalysis(domain=colliding_phantom)
        structural_analysis.set_measured_domains(self.registered_initial_brain, self.registered_deformed_brain)
        structural_analysis.set_boundary_condition(bc="fixture_3d_surface", referenced_plane="xz", direction="lower", threshold=-10)
        colliding_phantom_initial, colliding_phantom_fixed_nodes, colliding_phantom_deformed = structural_analysis.get_info()
        
        # constitutive model
        model = LinearElasticModel(domain=colliding_phantom)
        model.set_secondary_material(
            youngs_modulus=constants(category="plaplus_mechanical_properties", constant="youngs_modulus"),
            poissons_ratio=constants(category="plaplus_mechanical_properties", constant="poissons_ratio"),
            mass_density=constants(category="plaplus_mechanical_properties", constant="mass_density"),
            embedded_domain=lesion
        )
        
        # structural analysis: set constitutive model
        structural_analysis.set_material_model(constitutive_law=model)
        structural_analysis.set_initial_domain_orientation(goal_orientation=np.array([0, 0, -90]))
        if visualize: __visualize_initial()

        # structural analysis: construct global equation
        structural_analysis.construct_model()

        # structural analysis: apply gravity
        structural_analysis.set_boundary_condition(bc="gravity_3d_volume")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        # structural analysis: apply buoyancy
        structural_analysis.set_boundary_condition(bc="buoyancy_3d_surface", referenced_plane="xy", direction="lower", fluid_level=self.fluid_level, domain="deformed")
        structural_analysis.solve()
        structural_analysis.postprocessing()

        structural_analysis.save_report(saved_directory=self.saved_result_root_directory)
        structural_analysis.save_meshes(saved_directory=self.saved_mesh_root_directory)
        
        if visualize: __visualize_deformed()
            
        return structural_analysis, collided_phantom


class CuboidBrain3Experiment(CuboidBrain3):

    logger = TaskLogger("CB3EXPT")

    def __init__(self):
        super().__init__()

    def simulated_deformation_problem(self, visualize: bool = False):
        # this function wraps the problem that is needed to simulate;
        # thus, it is needed to be overwrite in subclass of Experiment4*
        return self.large_deformation_problem(visualize=visualize)

    def measurement_of_simulated_result(self, visualize: bool = True, save_screenshot: bool = False):
        
        self.logger.loginfo("Measurement of Simulated Result")
        
        # calculate the deformation
        structural_analysis, collided_phantom = self.simulated_deformation_problem(visualize=False)

        # save screenshot
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            directory=r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh",
            goal_orientation= np.array([90, 0, 0])
        )
        mpa.add_measured_meshes(
            domain_initial=self.registered_initial_brain,
            domain_deformed=self.registered_deformed_brain,
            embedded_domain_initial=self.registered_initial_lesion,
            embedded_domain_deformed=self.registered_deformed_lesion,
            # goal_orientation=np.array([0, 0, -90])
        )
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.visualize("save_screenshot", saved_screenshot_root_directory=self.saved_screenshot_root_directory, saved_graphic_root_directory=self.saved_graphic_root_directory)

    def result_analysis(self):
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            directory=r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh",
            goal_orientation= np.array([90, 0, 0])
        )
        mpa.add_measured_meshes(
            domain_initial=self.registered_initial_brain,
            domain_deformed=self.registered_deformed_brain,
            embedded_domain_initial=self.registered_initial_lesion,
            embedded_domain_deformed=self.registered_deformed_lesion,
            goal_orientation=np.array([0, 0, -90])
        )
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.result_analysis(self.saved_result_root_directory)

    def strains_analysis(self):
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            directory=r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh",
            goal_orientation= np.array([90, 0, 0])
        )
        mpa.add_measured_meshes(
            domain_initial=self.registered_initial_brain,
            domain_deformed=self.registered_deformed_brain,
            embedded_domain_initial=self.registered_initial_lesion,
            embedded_domain_deformed=self.registered_deformed_lesion,
            goal_orientation=np.array([0, 0, -90])
        )
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.strains_analysis()

    def equilibrium_residual_analysis(self, visualize: bool = True, save_screenshot: bool = False):
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            directory=r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh",
            goal_orientation= np.array([90, 0, 0])
        )
        mpa.add_measured_meshes(
            domain_initial=self.registered_initial_brain,
            domain_deformed=self.registered_deformed_brain,
            embedded_domain_initial=self.registered_initial_lesion,
            embedded_domain_deformed=self.registered_deformed_lesion,
            goal_orientation=np.array([0, 0, -90])
        )
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.equilibrium_residual_analysis(visualize=visualize, save_screenshot=save_screenshot, saved_screenshot_root_directory=self.saved_screenshot_root_directory, saved_graphic_root_directory=self.saved_graphic_root_directory)

    def save_screenshot(self):
        mpa = MeshPostAnalyzer()
        mpa.add_collided_domain(
            directory=r"data/phantom/cuboidbrain3/mesh/simulated/cuboidbrain3_skull.msh",
            goal_orientation= np.array([90, 0, 0])
        )
        mpa.add_measured_meshes(
            domain_initial=self.registered_initial_brain,
            domain_deformed=self.registered_deformed_brain,
            embedded_domain_initial=self.registered_initial_lesion,
            embedded_domain_deformed=self.registered_deformed_lesion,
            goal_orientation=np.array([0, 0, -90])
        )
        mpa.set_fluid_level(fluid_level=self.fluid_level)
        mpa.read_components(self.saved_mesh_root_directory)
        mpa.visualize("save_screenshot", saved_screenshot_root_directory=self.saved_screenshot_root_directory, saved_graphic_root_directory=self.saved_graphic_root_directory)


class Experiment4AWithContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4A")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = None

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4a_with_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="a")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_contact_problem(visualize=visualize)


class Experiment4AWithoutContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4A")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = None

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4a_without_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="a")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_problem(visualize=visualize)


class Experiment4BWithContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4B")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = 0

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4b_with_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="b")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_contact_buoyancy_contact_problem(visualize=visualize)


class Experiment4BWithoutContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4B")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = 0

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4b_without_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="b")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_buoyancy_contact_problem(visualize=visualize)


class Experiment4CWithContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4C")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = 75

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4c_with_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="c")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_contact_buoyancy_contact_problem(visualize=visualize)


class Experiment4CWithoutContact(CuboidBrain3Experiment):

    logger = TaskLogger("EXPT4C")

    def __init__(self):
        super().__init__()

        # overwrite parameters
        self.SOLVER = "sparse_cg"
        self.N_ITERATIONS_LIMIT = 10
        self.N_ITERATIONS_PATIENCE = 10
        self.PENETRATION_TOLERANCE = 0.1
        self.fluid_level = 75

        # directory
        self.result_root_directory = r"result/phantom/cuboidbrain3/expt4c_without_contact/"
        self.saved_result_root_directory = self.result_root_directory + r"error_analysis/"
        self.saved_screenshot_root_directory = self.result_root_directory + r"screenshots/"
        self.saved_graphic_root_directory = self.result_root_directory + r"graphics/"
        self.saved_mesh_root_directory = self.result_root_directory + r"mesh/"

        # segmented meshes registration
        self.segmented_meshes_registration(expt="c")

    def simulated_deformation_problem(self, visualize: bool = False):
        return self.large_deformation_buoyancy_contact_problem(visualize=visualize)


if __name__ == "__main__":
    # cuboidbrain3 = CuboidBrain3()
    # cuboidbrain3.segmented_meshes_registration(expt="a", visualize=False)
    # cuboidbrain3.large_deformation_problem()
    # cuboidbrain3.large_deformation_contact_problem()
    # cuboidbrain3.large_deformation_buoyancy_problem()
    # cuboidbrain3.large_deformation_contact_buoyancy_problem()
    # cuboidbrain3.test()

    expt4a = Experiment4AWithContact()
    # expt4a.measurement_of_simulated_result(visualize=False, save_screenshot=False)
    # expt4a.result_analysis()
    expt4a.save_screenshot()
    # expt4a.strains_analysis()
    # expt4a.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    del expt4a
    # exit()

    expt4b = Experiment4BWithContact()
    # expt4b.measurement_of_simulated_result(visualize=False, save_screenshot=False)
    # expt4b.result_analysis()
    expt4b.save_screenshot()
    # expt4b.strains_analysis()
    # expt4b.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    del expt4b
    # exit()

    expt4c = Experiment4CWithContact()
    # expt4c.measurement_of_simulated_result(visualize=False, save_screenshot=False)
    # expt4c.result_analysis()
    expt4c.save_screenshot()
    # expt4c.strains_analysis()
    # expt4c.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    del expt4c
    # exit()

    # expt4a = Experiment4AWithoutContact()
    # expt4a.measurement_of_simulated_result(visualize=False, save_screenshot=True)
    # expt4a.result_analysis()
    # expt4a.save_screenshot()
    # expt4a.strains_analysis()
    # expt4a.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    # del expt4a
    # exit()

    # expt4b = Experiment4BWithoutContact()
    # expt4b.measurement_of_simulated_result(visualize=False, save_screenshot=True)
    # expt4b.result_analysis()
    # expt4b.save_screenshot()
    # expt4b.strains_analysis()
    # expt4b.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    # del expt4b
    # exit()

    # expt4c = Experiment4CWithoutContact()
    # expt4c.measurement_of_simulated_result(visualize=False, save_screenshot=True)
    # expt4c.result_analysis()
    # expt4c.save_screenshot()
    # expt4c.strains_analysis()
    # expt4c.equilibrium_residual_analysis(visualize=False, save_screenshot=True)
    # del expt4c
    # exit()
