# Brain Shift Prediction

FEM simulation for brain shift prediction.


## Package

The developed python package is contained under `src/`.
- `/core`: Core package, which are FEM and contact mechanics.
- `/solver`: Linear algebra solver.
- `/utils`: Utilities for helping in validation experiments, postprocessing, etc.
- `/config`: Parameters and constants in `.yaml`.


## Scripts

The uploaded scripts are the scripts to run for the validation experiments and demonstration.
- `phantom_cuboidbrain3.py`: The main validation experiment using phantom.
- `patient_mrbraintumor1.py`: The demonstration with clinical data.

The script can be directly run. In the Python script, modify the code under `__main__` to execute the desired process.

**Example 1** Validation experiments with phantom
```
expt4a = Experiment4AWithContact()
expt4a.measurement_of_simulated_result()
```

**Example 2** Demonstration with clinical data
```
demo1a = Demonstration1A()
demo1a.demonstration_of_brain_shift_result()
```

## Data

The uploaded data include:
- `cuboidbrain3`: A geometric silicone brain phantom, scanned with CT scanner. This phantom is used for the validation experiments.
- `mrbraintumor1`: An MR images of brain with tumor, obtained from example dataset of `3D Slicer`. This data are used for demonstration purpose.

The current uploaded data are the main data used in the study, which compose of the 3D reconstructed of segmented images and its generated meshes in `Gmsh` format. In the study, there are more phantom data used in the validation. Due to the file size and number of data, it cannot uploaded into GitHub repository, please contact the author to request the rest of data.


## Contact
Tanadul Somboonwong

Email: tanadul.sob@student.mahidol.ac.th

Center for Biomedical and Robotics Technology (BART LAB), Department of Biomedical Engineering, Faculty of Engineering, Mahidol University, Thailand

