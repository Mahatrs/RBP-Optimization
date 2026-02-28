# Genetic Algorithm – Robot Base Placement Optimization

## Description

This module implements a Genetic Algorithm (GA) to optimize the base position of a robotic manipulator.

The algorithm searches for the optimal base coordinates (X, Y, Z) that maximize a weighted kinematic performance score based on:

- Manipulability
- Distance from joint limits
- Singularity robustness

The robot model is loaded from a URDF file (`iiwa14.urdf`).

The script performs multiple GA runs and exports:
- CSV results
- Statistical summary file

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- IKPy
- Matplotlib
- A valid URDF robot model

Install dependencies with:
```
pip install numpy scipy pandas matplotlib ikpy
```
---

## What to Modify for Your Own Robot

If you want to adapt this code to another robot or application, you must modify:

1. **URDF file**
   - Replace `iiwa14.urdf` with your robot’s URDF.
   - Update the base link name if needed.

2. **Joint limits**
   - Update the `JOINT_LIMITS` array in `ga_data.py`.

3. **Workspace bounds**
   - Modify the `bounds` variable:
     ```
     bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
     ```

4. **Objective weights**
   - Adjust metric weights inside the objective function if needed.

5. **Jacobian model (if required)**
   - If using a different robot, the analytical Jacobian may need to be updated.

---

## Notes

The performance metrics are kinematic-based and task-dependent.  
The objective function should be adapted to reflect the requirements of your specific application.
