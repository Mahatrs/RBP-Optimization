# Cuboid-Based Workspace Robustness Evaluation

## Description

This module evaluates the kinematic performance of a robotic manipulator inside a bounded 3D workspace.

It:

1. Generates a 3D spatial grid around a target.
2. Solves inverse kinematics at each grid point.
3. Computes performance metrics.
4. Produces a normalized 3D heatmap.

The robot model is loaded from a URDF file (`iiwa14.urdf`).

The script generates a 3D visualization of the evaluated cuboid workspace.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- IKPy
- A valid URDF robot model

Install dependencies with:
```
pip install numpy scipy matplotlib ikpy
```
---

## What to Modify for Your Own Robot

To adapt this module:

1. **URDF file**
   - Replace `iiwa14.urdf` with your robot’s URDF.
   - Update the base link name if necessary.

2. **Target position**
   - Modify the `TARGET` variable in `main.py`.

3. **Workspace dimensions**
   - Adjust:
     ```
     RADIUS_X
     RADIUS_Y
     RADIUS_Z
     STEP
     ```

4. **Orientation sampling**
   - Modify:
     ```
     AXIS7
     HALF_ANGLE
     N_ORIENT
     ```

5. **Joint limits**
   - Update the `JOINT_LIMITS` definition in `robot_utils.py`.

6. **Jacobian implementation**
   - If using a different robot, update the analytical Jacobian in `robot_utils.py`.

---

## Notes

The cuboid evaluation assumes local kinematic feasibility.  
For different applications, workspace definition and metric weights should be adapted accordingly.
