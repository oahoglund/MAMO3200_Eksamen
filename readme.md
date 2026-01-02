# Double-Slit Simulation (MAMO3200 Exam Project)

This repository contains my exam project for **MAMO3200**, where I model and simulate the **quantum mechanical double-slit experiment** using numerical methods.

The project solves the **time-dependent Schrödinger equation** for a particle passing through a double-slit potential, starting from a Gaussian wave packet. The time evolution is computed using several numerical schemes, and the results are visualized using both static plots and 3D animations.

As the course was in Norwegian, the code and presentation is written in Norwegian.

---

## Physical Model

- **System**: 2D quantum particle in a box  
- **Initial condition**: Gaussian wave packet with momentum in the x-direction  
- **Potential**: High potential barrier forming a **double slit**  
- **Boundary conditions**: Dirichlet boundary conditions  
  $$
  \psi = 0 \quad \text{on the boundary}
  $$

The Hamiltonian is discretized using finite differences, resulting in a sparse matrix representation.

---

## Numerical Methods

The following time-stepping methods are implemented and compared:

### Explicit methods
- Forward Finite Difference
- Leapfrog
- Runge–Kutta 4 (RK4)

These methods are analyzed with respect to **stability** and **probability conservation**.

### Implicit method
- Crank–Nicolson

The Crank–Nicolson scheme is unconditionally stable and serves as a reference solution.

The total probability  
$$
\int |\psi(x,y,t)|^2 \, dx \, dy
$$
is monitored over time to assess numerical accuracy and stability.

---

## 🧑‍💻 Code Structure

```
.
├── main.py
├── simulering.py
├── visualisering.py
├── presentasjon.ipynb
```

### `simulering.py`
Contains all simulation logic:
- Initialization of the Gaussian wave packet
- Construction of the Hamiltonian using sparse matrices
- Time integration schemes
- Normalization and probability diagnostics

### `visualisering.py`
Handles visualization:
- 3D surface plots using Mayavi
- Animations of the probability density
- Plots of total probability over time
- GIF creation using `ffmpeg`

### `main.py`
Acts as the entry point:
- Connects simulation and visualization
- Contains test functions for experimenting with parameters and numerical methods

### `presentasjon.ipynb`
A Jupyter Notebook used during the exam to present:
- The physical setup
- Numerical results
- Stability comparisons between methods

---

## Visual Output

The project supports:
- Static 3D snapshots of the probability density
- Real-time animations
- Saved GIF animations
- Probability conservation plots

---

## 🛠 Requirements

Main dependencies:
- `numpy`
- `scipy`
- `matplotlib`
- `mayavi`
- `ffmpeg` (required for GIF export)

Install:
```bash
pip install requirements.txt
```

> **Note:** Mayavi is recommended to be installed via **conda**, especially on Windows.

---

## Running the Project

Activate your environment and run:

```bash
python main.py
```

Different test functions can be uncommented in `main.py` to:
- Switch numerical methods
- Generate animations
- Save figures and GIFs

---

## Skills Demonstrated

This project demonstrates:
- Numerical solution of partial differential equations
- Stability analysis of time-integration methods
- Sparse linear algebra
- Scientific Python programming
- Modular code design
- Advanced visualization techniques
- Reproducible computational workflows

---

## 📌 Notes

This project was developed as part of an exam, but the codebase is written with clarity, modularity, and reproducibility in mind. It is intended both as a scientific simulation and as a demonstration of numerical and computational physics skills.
