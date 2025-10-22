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

# Build absolute timestamps from Timestamp + Milliseconds
df["abs_time"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S") \
                   + pd.to_timedelta(df["milliseconds"], unit="ms")
timestamps = (df["abs_time"] - df["abs_time"].iloc[0]).dt.total_seconds().to_numpy()

# Raw sensors
acc_raw = df[["accx", "accy", "accz"]].to_numpy().astype(np.float32)
gyro_raw = df[["gyrox", "gyroy", "gyroz"]].to_numpy().astype(np.float32)  # likely deg/s

# =========================
# FRAME REMAPPING (Bike → Model)
# Bike:  x=down, y=left, z=forward
# Model: x=forward, y=right, z=down
# =========================
MAPPING_OPTION = 1

if MAPPING_OPTION == 1:
    print("Using mapping option 1: Standard (yaw inverted)")
    acc = np.column_stack([acc_raw[:, 2], -acc_raw[:, 1], acc_raw[:, 0]])
    gyro = np.column_stack([gyro_raw[:, 2], -gyro_raw[:, 1], -gyro_raw[:, 0]])  # deg/s
elif MAPPING_OPTION == 2:
    print("Using mapping option 2: Z-up convention")
    acc = np.column_stack([acc_raw[:, 2], -acc_raw[:, 1], -acc_raw[:, 0]])
    gyro = np.column_stack([gyro_raw[:, 2], -gyro_raw[:, 1], -gyro_raw[:, 0]])
elif MAPPING_OPTION == 3:
    print("Using mapping option 3: No y-axis flip")
    acc = np.column_stack([acc_raw[:, 2], acc_raw[:, 1], acc_raw[:, 0]])
    gyro = np.column_stack([gyro_raw[:, 2], gyro_raw[:, 1], -gyro_raw[:, 0]])
elif MAPPING_OPTION == 4:
    print("Using mapping option 4: 180° rotation")
    acc = np.column_stack([-acc_raw[:, 2], acc_raw[:, 1], acc_raw[:, 0]])
    gyro = np.column_stack([-gyro_raw[:, 2], gyro_raw[:, 1], -gyro_raw[:, 0]])

print("\nFrame remapping check:")
print("acc sample (before norm):", acc_raw[0])
print("acc mapped (bike→model):", acc[0])
print("gyro sample (deg/s):", gyro_raw[0])
print("gyro mapped (deg/s):", gyro[0])

# Normalize accelerometer (safe-divide)
norm = np.linalg.norm(acc, axis=1, keepdims=True)
norm[norm == 0] = 1.0
acc_normalized = acc / norm

# Gyro to rad/s for all downstream integration
gyro_rad = np.deg2rad(gyro)

# =========================
# CREATE OVERLAPPING WINDOWS (+ keep indices)
# =========================
N = len(acc_normalized)
idxs = list(range(0, N - window_size + 1, stride))  # window start indices

def stack_windows(arr, starts, w):
    return np.stack([arr[s:s + w] for s in starts], axis=0)

acc_windows = stack_windows(acc_normalized, idxs, window_size)
gyro_windows = stack_windows(gyro_rad,       idxs, window_size)

# Per-window effective sampling frequency (for model input)
# fs ≈ (window_size - 1) / duration_of_window
win_durations = np.array([
    max(timestamps[s + window_size - 1] - timestamps[s], 1e-6) for s in idxs
], dtype=np.float64)
fs_per_window = (window_size - 1) / win_durations
fs_input = fs_per_window.reshape(-1, 1).astype(np.float32)

print(f"\nPrepared {len(acc_windows)} windows of shape {acc_windows.shape}")
print(f"Mean fs from data: {fs_per_window.mean():.2f} Hz")

# End time of each window (align predictions to these times)
times = np.array([timestamps[s + window_size - 1] for s in idxs], dtype=np.float64)

# =========================
# MODEL INFERENCE
# =========================
print("\nRunning model inference...")
pred_quats = model.predict([acc_windows, gyro_windows, fs_input], verbose=1)
np.savetxt("predicted_quaternions2.csv", pred_quats, delimiter=",", fmt="%.6f")
print("✅ Predictions saved to predicted_quaternions.csv")

# =========================
# DIAGNOSTICS
# =========================
print("\n=== MODEL OUTPUT DIAGNOSTICS ===")
print("Example quaternion (first window):", pred_quats[0])

quat_std = np.std(pred_quats, axis=0)
print("Quaternion std (w,x,y,z):", quat_std)

angles = 2 * np.arccos(np.clip(pred_quats[:, 0], -1.0, 1.0))
print(f"Rotation angles - mean: {np.degrees(angles.mean()):.2f}°, "
      f"max: {np.degrees(angles.max()):.2f}°, "
      f"min: {np.degrees(angles.min()):.2f}°")

norms = np.linalg.norm(pred_quats, axis=1)
print(f"Quaternion norms - mean: {norms.mean():.6f}, "
      f"min: {norms.min():.6f}, max: {norms.max():.6f}")

# =========================
# QUATERNION → EULER (rad)
# =========================
def quat_to_euler(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - x*z), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

roll_pred, pitch_pred, yaw_pred = quat_to_euler(pred_quats)

# =========================
# CALIBRATION (bias removal)
# =========================
CALIBRATION_SAMPLES = min(10, len(roll_pred))
roll_offset  = np.mean(roll_pred[:CALIBRATION_SAMPLES])
pitch_offset = np.mean(pitch_pred[:CALIBRATION_SAMPLES])

print("\n=== CALIBRATION OFFSETS ===")
print(f"Roll offset:  {np.degrees(roll_offset):.2f}°")
print(f"Pitch offset: {np.degrees(pitch_offset):.2f}°")

roll_pred  = roll_pred  - roll_offset
pitch_pred = pitch_pred - pitch_offset

print(f"\nModel yaw range: {np.degrees(yaw_pred.min()):.2f}° to {np.degrees(yaw_pred.max()):.2f}°")
print(f"Roll range: {np.degrees(roll_pred.min()):.2f}° to {np.degrees(roll_pred.max()):.2f}°")
print(f"Pitch range: {np.degrees(pitch_pred.min()):.2f}° to {np.degrees(pitch_pred.max()):.2f}°")

# =========================
# INTEGRATE GYRO FOR YAW (rad) — aligned to window ends
# =========================
print("\n=== INTEGRATING YAW FROM GYROSCOPE ===")

num_windows = len(idxs)
yaw_integrated = np.zeros(num_windows, dtype=np.float64)  # radians

# End indices for each prediction window
end_indices = np.array([s + window_size - 1 for s in idxs], dtype=int)

for i in range(1, num_windows):
    prev_end = end_indices[i - 1]
    curr_end = end_indices[i]
    start = prev_end + 1
    end = curr_end + 1  # slice end-exclusive

    if start >= end or end > len(gyro_rad):
        # Fallback: use a single-sample dt if something edge-casey happens
        dt_step = max(times[i] - times[i - 1], 1e-6)
        gyro_z_avg = gyro_rad[min(curr_end, len(gyro_rad) - 1), 2]
    else:
        dt_step = timestamps[end - 1] - timestamps[start - 1]
        dt_step = max(dt_step, 1e-9)
        gyro_z_avg = np.mean(gyro_rad[start:end, 2])  # rad/s

    yaw_integrated[i] = yaw_integrated[i - 1] + gyro_z_avg * dt_step  # radians

print(f"Integrated yaw range: {np.degrees(yaw_integrated.min()):.2f}° to {np.degrees(yaw_integrated.max()):.2f}°")
print(f"Total yaw change:     {np.degrees(yaw_integrated[-1] - yaw_integrated[0]):.2f}°")

# =========================
# SAVE RESULTS (degrees)
# =========================
results = np.column_stack([
    times,
    np.degrees(yaw_integrated),
    np.degrees(roll_pred),
    np.degrees(pitch_pred),
])
np.savetxt(
    "predicted_orientation2.csv",
    results,
    delimiter=",",
    fmt="%.4f",
    header="time(s),yaw(deg),roll(deg),pitch(deg)",
    comments="",
)
print("\n✅ Full orientation saved to predicted_orientation.csv")

np.savetxt(
    "predicted_yaw2.csv",
    np.column_stack([times, np.degrees(yaw_integrated)]),
    delimiter=",",
    fmt="%.4f",
    header="time(s),yaw(deg)",
    comments="",
)
print("✅ Yaw angles saved to predicted_yaw.csv")

# =========================
# BASIC MOTION ANALYSIS
# =========================
print("\n=== MOTION ANALYSIS ===")
if len(times) >= 2:
    dt_steps = np.diff(times)
    dt_steps = np.where(dt_steps <= 0, 1e-6, dt_steps)
    yaw_changes = np.diff(yaw_integrated)  # rad
    yaw_rate = yaw_changes / dt_steps      # rad/s

    print("Yaw rate stats:")
    print(f"  Mean: {np.degrees(np.mean(yaw_rate)):.2f}°/s")
    print(f"  Max:  {np.degrees(np.max(yaw_rate)):.2f}°/s")
    print(f"  Min:  {np.degrees(np.min(yaw_rate)):.2f}°/s")

    turn_threshold_deg = 2.0
    turn_threshold_rad = np.deg2rad(turn_threshold_deg)
    left_turns  = np.sum(yaw_changes >  turn_threshold_rad)
    right_turns = np.sum(yaw_changes < -turn_threshold_rad)
    print(f"\nTurn detection (>{turn_threshold_deg:.1f}°/step):")
    print(f"  Left turns:  {left_turns}")
    print(f"  Right turns: {right_turns}")
else:
    print("Not enough samples for motion analysis.")
# =========================
# YAW RATE CLASSIFICATION (deg/s)
# =========================
print("\n=== YAW RATE CLASSIFICATION ===")

# unwrap integrated yaw (radians → degrees)
yaw_deg_int = np.degrees(yaw_integrated)
yaw_deg_int_unwrapped = np.degrees(np.unwrap(np.radians(yaw_deg_int), discont=np.pi))

# deltas and dt
yaw_deltas = np.diff(yaw_deg_int_unwrapped)  # deg
dt_steps = np.diff(times)                    # sec
dt_steps = np.where(dt_steps <= 0, 1e-6, dt_steps)

yaw_rate = yaw_deltas / dt_steps             # deg/s
yaw_times = times[1:]

def classify(rate):
    mag = abs(rate)
    if mag < 2.0:
        return "straight"
    elif mag < 5.0:
        return "slight right" if rate > 0 else "slight left"
    elif mag < 15.0:
        return "normal right" if rate > 0 else "normal left"
    else:
        return "major right" if rate > 0 else "major left"

labels = [classify(r) for r in yaw_rate]

# Save
out_df = pd.DataFrame({
    "time(s)": yaw_times,
    "delta_yaw(deg)": yaw_deltas,
    "dt(s)": dt_steps,
    "yaw_rate(deg/s)": yaw_rate,
    "turn_type": labels
})
out_df.to_csv("yaw_rate_classified.csv", index=False, float_format="%.6f")

print("✅ Saved yaw_rate_classified.csv")
print("Stats: mean={:.2f}°/s, max={:.2f}°/s, min={:.2f}°/s".format(
    yaw_rate.mean(), yaw_rate.max(), yaw_rate.min()
))