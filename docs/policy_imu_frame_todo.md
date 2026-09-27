# TODO: align IMU observations across policies

The immediate PICO v12 frame mismatch has been corrected in the robot runtime:
the v12 actor receives raw BMI088 gyro values in the IMU site's local axes,
matching Mjlab's `robot/imu_ang_vel` observation and the v12 ONNX metadata
`base_ang_vel_frame=imu_sensor_xyz`. This correction does not change the v12
checkpoint or ONNX action weights. Broader frame unification remains follow-up
work after the get-up retraining milestone.

The TWIST2 tracking actor, velocity walking actor, get-up actor, and PICO v12
training actor read `robot/imu_ang_vel` in the IMU site's local axes. The robot
feeds raw BMI088 gyro values to walking, get-up, and now PICO v12. The historical
PICO v10 contract still uses `sensor_gyro_to_body` and `robot_body_xyz`. A
separate train/export/runtime audit should verify each remaining model's exact
frame and document the intended common convention.

For broader unification:

1. Check the physical BMI088 axis signs against the MJCF IMU site and the
   recorded actor observation for TWIST2, walking, and each teleop recipe.
2. Pick one explicit observation frame per model. Update its training config,
   checkpoint provenance, ONNX metadata, and robot observation builder together.
3. Export a newly authenticated ONNX, evaluate it in simulation, and verify the
   actual robot behavior before replacing the working artifact.

The get-up v2 contract already requires `imu_sensor_xyz`. Its retraining and
deployment can proceed independently of this TODO.
