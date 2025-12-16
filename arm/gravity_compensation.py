import numpy as np

# Constants
g = 9.81  # gravitational acceleration in m/s^2

# Link properties
m2 = 0.58247  # mass of link 2 in kg

# The original position of the COM of link 2 (before rotation)
x2, y2, z2 = 0.021945, 0.098953, -7.9847E-08  # COM of link 2 from URDF

# Rotation matrix for 90 degrees around the x-axis
rotation_matrix_x_90 = np.array([[1, 0, 0],
                                 [0, 0, -1],
                                 [0, 1, 0]])

# Original position vector (r) from joint 2 to the COM of link 2
r = np.array([x2, y2, z2])

# Apply the rotation to the position vector r
r_rot = np.dot(rotation_matrix_x_90, r)

# Extract the new coordinates after rotation
x2_rot, y2_rot, z2_rot = r_rot[0], r_rot[1], r_rot[2]

# Gravitational force on link 2 (force due to gravity, in the negative z-direction)
F_gravity = np.array([0, 0, -m2 * g])  # gravity force in Newtons

# Calculate the torque due to gravity at joint 2 using the cross product
torque_gravity = np.cross(r_rot, F_gravity)

# Output the new position of the COM after rotation and the gravity compensation torque
print("Rotated COM of link 2:", r_rot)
print("Gravity compensation torque at joint 2:", torque_gravity)


# Joint 2 axis (unit vector)
v_joint_2 = np.array([0, 0.5, 0.5]) # A.dot(u)*u

# Normalize the joint axis (in case it's not a unit vector, though it's assumed here)
v_joint_2 = v_joint_2 / np.linalg.norm(v_joint_2)

# Calculate the projection of torque onto the joint axis
torque_projection = np.dot(torque_gravity, v_joint_2) * v_joint_2

# Output the result
print("Projection of gravity compensation torque onto joint 2's axis:", torque_projection)

