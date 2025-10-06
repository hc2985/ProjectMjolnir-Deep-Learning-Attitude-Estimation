import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from model import Model_A


# =========================
# CONFIGURATION
# =========================
window_size = 100
stride = 10
fs_value = 50  # Hz
weights_filename = "Model_A_B500_E300_V2.hdf5"
data_filename = "data.csv"

# =========================
# LOAD MODEL AND WEIGHTS
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, weights_filename)

model = Model_A(window_size)
model.load_weights(weights_path, by_name=True, skip_mismatch=False)
print("✅ Model and weights loaded")

# =========================
# LOAD AND PREPROCESS DATA
# =========================
data_path = os.path.join(script_dir, data_filename)
df = pd.read_csv(data_path)
df.columns = [c.strip().lower() for c in df.columns]
print("Detected columns:", df.columns.tolist())

acc_raw = df[["accx", "accy", "accz"]].to_numpy().astype(np.float32)
gyro_raw = df[["gyrox", "gyroy", "gyroz"]].to_numpy().astype(np.float32)

# =========================
# FRAME REMAPPING (Bike → Model)
# =========================
# Bike frame:  x = down, y = left, z = forward
# Model frame: x = forward, y = right, z = down

MAPPING_OPTION = 1  # Try 1–4 if alignment looks wrong

if MAPPING_OPTION == 1:
    print("Using mapping option 1: Standard (yaw inverted)")
    acc = np.column_stack([
        acc_raw[:, 2],      # forward
        -acc_raw[:, 1],     # right
        acc_raw[:, 0]       # down
    ])
    gyro = np.column_stack([
        gyro_raw[:, 2],
        -gyro_raw[:, 1],
        -gyro_raw[:, 0]
    ])

elif MAPPING_OPTION == 2:
    print("Using mapping option 2: Z-up convention")
    acc = np.column_stack([
        acc_raw[:, 2],
        -acc_raw[:, 1],
        -acc_raw[:, 0]
    ])
    gyro = np.column_stack([
        gyro_raw[:, 2],
        -gyro_raw[:, 1],
        -gyro_raw[:, 0]
    ])

elif MAPPING_OPTION == 3:
    print("Using mapping option 3: No y-axis flip")
    acc = np.column_stack([
        acc_raw[:, 2],
        acc_raw[:, 1],
        acc_raw[:, 0]
    ])
    gyro = np.column_stack([
        gyro_raw[:, 2],
        gyro_raw[:, 1],
        -gyro_raw[:, 0]
    ])

elif MAPPING_OPTION == 4:
    print("Using mapping option 4: 180° rotation")
    acc = np.column_stack([
        -acc_raw[:, 2],
        acc_raw[:, 1],
        acc_raw[:, 0]
    ])
    gyro = np.column_stack([
        -gyro_raw[:, 2],
        gyro_raw[:, 1],
        -gyro_raw[:, 0]
    ])

# =========================
# DEBUG PRINTS
# =========================
print("\nFrame remapping check:")
print("acc sample (before norm):", acc_raw[0])
print("acc mapped (bike→model):", acc[0])
print("gyro sample (deg/s):", gyro_raw[0])
print("gyro mapped (before rad/s conversion):", gyro[0])

# Normalize accelerometer and convert gyro to radians/sec
acc_normalized = acc / np.linalg.norm(acc, axis=1, keepdims=True)
gyro_rad = np.deg2rad(gyro)

# =========================
# CREATE OVERLAPPING WINDOWS
# =========================
def make_windows(arr, window_size, step):
    N = len(arr)
    idxs = range(0, N - window_size + 1, step)
    return np.stack([arr[i:i + window_size] for i in idxs], axis=0)

acc_windows = make_windows(acc_normalized, window_size, stride)
gyro_windows = make_windows(gyro_rad, window_size, stride)
fs_input = np.full((len(acc_windows), 1), fs_value)

print(f"\nPrepared {len(acc_windows)} windows of shape {acc_windows.shape}")

# =========================
# MODEL INFERENCE
# =========================
print("\nRunning model inference...")
pred_quats = model.predict([acc_windows, gyro_windows, fs_input], verbose=1)
np.savetxt("predicted_quaternions.csv", pred_quats, delimiter=",", fmt="%.6f")
print("✅ Predictions saved to predicted_quaternions.csv")

# =========================
# DIAGNOSTICS
# =========================
print("\n=== MODEL OUTPUT DIAGNOSTICS ===")
print("Example quaternion (first window):", pred_quats[0])

quat_std = np.std(pred_quats, axis=0)
print("Quaternion std (w,x,y,z):", quat_std)

angles = 2 * np.arccos(np.clip(pred_quats[:, 0], -1, 1))
print(f"Rotation angles - mean: {np.degrees(angles.mean()):.2f}°, "
      f"max: {np.degrees(angles.max()):.2f}°, "
      f"min: {np.degrees(angles.min()):.2f}°")

norms = np.linalg.norm(pred_quats, axis=1)
print(f"Quaternion norms - mean: {norms.mean():.6f}, "
      f"min: {norms.min():.6f}, max: {norms.max():.6f}")

# =========================
# CONVERT QUATERNIONS TO EULER ANGLES
# =========================
def quat_to_euler(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - x*z), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

roll_pred, pitch_pred, yaw_pred = quat_to_euler(pred_quats)

# =========================
# CALIBRATION
# =========================
CALIBRATION_SAMPLES = 10
roll_offset = np.mean(roll_pred[:CALIBRATION_SAMPLES])
pitch_offset = np.mean(pitch_pred[:CALIBRATION_SAMPLES])

print("\n=== CALIBRATION OFFSETS ===")
print(f"Roll offset: {np.degrees(roll_offset):.2f}°")
print(f"Pitch offset: {np.degrees(pitch_offset):.2f}°")

roll_pred -= roll_offset
pitch_pred -= pitch_offset

print(f"\nModel yaw range: {np.degrees(yaw_pred.min()):.2f}° to {np.degrees(yaw_pred.max()):.2f}°")
print(f"Roll range: {np.degrees(roll_pred.min()):.2f}° to {np.degrees(roll_pred.max()):.2f}°")
print(f"Pitch range: {np.degrees(pitch_pred.min()):.2f}° to {np.degrees(pitch_pred.max()):.2f}°")

# =========================
# INTEGRATE GYRO FOR YAW
# =========================
print("\n=== INTEGRATING YAW FROM GYROSCOPE ===")

yaw_integrated = np.zeros(len(pred_quats))
dt = stride / fs_value

for i in range(1, len(pred_quats)):
    start = i * stride
    end = min(start + stride, len(gyro))
    gyro_z_avg = np.mean(gyro[start:end, 2])
    yaw_integrated[i] = yaw_integrated[i - 1] + gyro_z_avg * dt

print(f"Integrated yaw range: {yaw_integrated.min():.2f}° to {yaw_integrated.max():.2f}°")
print(f"Total yaw change: {yaw_integrated[-1] - yaw_integrated[0]:.2f}°")

# =========================
# COMBINE RESULTS
# =========================
times = np.arange(len(pred_quats)) * dt

print("\n=== FINAL ORIENTATION SAMPLE ===")
for i in range(min(10, len(times))):
    print(f"t={times[i]:6.2f}s | yaw={yaw_integrated[i]:8.2f}° | "
          f"roll={np.degrees(roll_pred[i]):8.2f}° | pitch={np.degrees(pitch_pred[i]):8.2f}°")
if len(times) > 10:
    print("...")
    for i in range(max(len(times) - 3, 10), len(times)):
        print(f"t={times[i]:6.2f}s | yaw={yaw_integrated[i]:8.2f}° | "
              f"roll={np.degrees(roll_pred[i]):8.2f}° | pitch={np.degrees(pitch_pred[i]):8.2f}°")

# =========================
# SAVE RESULTS
# =========================
results = np.column_stack([times, yaw_integrated,
                           np.degrees(roll_pred), np.degrees(pitch_pred)])
np.savetxt("predicted_orientation.csv", results, delimiter=",", fmt="%.4f",
           header="time(s),yaw(deg),roll(deg),pitch(deg)", comments="")
print("\n✅ Full orientation saved to predicted_orientation.csv")

np.savetxt("predicted_yaw.csv",
           np.column_stack([times, yaw_integrated]),
           delimiter=",", fmt="%.4f",
           header="time(s),yaw(deg)", comments="")
print("✅ Yaw angles saved to predicted_yaw.csv")

# =========================
# BASIC MOTION ANALYSIS
# =========================
print("\n=== MOTION ANALYSIS ===")
yaw_changes = np.diff(yaw_integrated)

if len(yaw_changes):
    print("Yaw rate stats:")
    print(f"  Mean: {np.mean(yaw_changes) / dt:.2f}°/s")
    print(f"  Max: {np.max(yaw_changes) / dt:.2f}°/s")
    print(f"  Min: {np.min(yaw_changes) / dt:.2f}°/s")

    turn_threshold = 2.0
    left_turns = np.sum(yaw_changes > turn_threshold)
    right_turns = np.sum(yaw_changes < -turn_threshold)
    print(f"\nTurn detection (>{turn_threshold}°/step):")
    print(f"  Left turns: {left_turns}")
    print(f"  Right turns: {right_turns}")
