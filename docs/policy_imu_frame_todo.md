# TODO: align IMU observations across policies

This is follow-up work **after** the get-up retraining milestone. Keep the
currently working walking and PICO control paths, their ONNX files, and their
runtime observation builders unchanged for that milestone.

The TWIST2 tracking actor, velocity walking actor, and get-up actor read
`robot/imu_ang_vel` in the IMU site's local axes. The robot feeds the raw BMI088
gyro values to walking and get-up. PICO's current runtime rotates those values
with `sensor_gyro_to_body` and labels the ONNX input `robot_body_xyz`; the Mjlab
teleop actor inherits the raw sensor observation. This deserves a separate
train/export/runtime audit even though the current PICO controls are usable.

Before changing the deployed PICO path:

1. Check the physical BMI088 axis signs against the MJCF IMU site and the
   recorded actor observation for TWIST2, walking, and each teleop recipe.
2. Pick one explicit observation frame per model. Update its training config,
   checkpoint provenance, ONNX metadata, and robot observation builder together.
3. Export a newly authenticated ONNX, evaluate it in simulation, and verify the
   actual robot behavior before replacing the working artifact.

The get-up v2 contract already requires `imu_sensor_xyz`. Its retraining and
deployment can proceed independently of this TODO.
