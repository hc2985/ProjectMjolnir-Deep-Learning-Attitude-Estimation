# Project Mjolnir – IMU Orientation Estimation (6DoF)

This repository is part of **Project Mjolnir**, an adaptive mountain bike telemetry system designed to analyze rider and suspension dynamics.  
It adapts the deep learning framework from  
**“Generalizable end-to-end deep learning frameworks for real-time attitude estimation using 6DoF inertial measurement units”**  
(*Measurement*, 2023) to process and interpret IMU data collected from a **bike-mounted telemetry system**.

Original framework: [Armanasq/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU](https://github.com/Armanasq/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU)  
Paper DOI: [10.1016/j.measurement.2023.113105](https://doi.org/10.1016/j.measurement.2023.113105)

---

## Overview

This implementation focuses on **inference and analysis**, not retraining.  
It uses a pretrained model to estimate quaternions and derived orientation angles (**roll**, **pitch**, and **yaw**) from **6DoF IMU data** — accelerometer and gyroscope only.

The system is tailored for **bike-mounted IMUs** placed on the frame and rider, where axes differ from the dataset used in the original paper.  
A configurable frame remapping layer is added to align input data with the model’s expected coordinate frame.

---

## Key Features

- Uses pretrained model from the original research  
- Processes IMU CSV data from Project Mjolnir’s sensors  
- Supports flexible frame remapping (bike ↔ model)  
- Outputs predicted **quaternions**, **roll**, **pitch**, and **yaw**  
- Integrates yaw from gyroscope data to correct drift  
- Automatically applies roll/pitch calibration using initial stationary samples  
- Saves detailed CSV outputs for further analysis  

---

## Repository Structure

| File | Description |
|------|--------------|
| `model.py` | Defines the model architecture (same as the original paper). |
| `processdata.py` | Main inference script — handles data loading, preprocessing, quaternion prediction, and yaw integration. |
| `data.csv` | Example telemetry data (acceleration and gyro readings). |
| `Model_A_B500_E300_V2.hdf5` | Pretrained model weights from the published paper. |

---

## Coordinate Frames

**Bike-mounted IMU (input)**  
x = Down  
y = Left  
z = Forward  

**Model’s expected frame (training convention)**  
x = Forward  
y = Right  
z = Down  

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- TensorFlow Addons  
- NumPy  
- Pandas  

---

## Citation

If you use or reference this implementation, please cite the original paper:

@article{GOLROUDBARI2023113105,  
title = {Generalizable end-to-end deep learning frameworks for real-time attitude estimation using 6DoF inertial measurement units},  
journal = {Measurement},  
pages = {113105},  
year = {2023},  
doi = {10.1016/j.measurement.2023.113105},  
author = {Arman Asgharpoor Golroudbari and Mohammad Hossein Sabour}  
}

---

## License

This project inherits the **MIT License** from the original repository.
