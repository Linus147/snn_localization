# Real-Time Object Localization with a Hardware-Coupled Spiking Neural Network (ALIF)

This project connects two HC-SR04 ultrasonic sensors and an 8x32 MAX7219 LED matrix on a Raspberry Pi 5 to a Spiking Neural Network (SNN) written from scratch in pure NumPy. The goal is to localize an object across a 40x20 cm grid (8x4 cells) using only the time-of-flight measurements from both sensors.

The network operates as a continuous attractor system: each of the 128 excitatory neurons drives a single physical LED on the matrix. Which grid cell lights up is determined entirely by the internal dynamics of local recurrent excitation and global inhibition across populations.

---

## Hardware Setup and Calibration

The physical setup consists of:
* Raspberry Pi 5
* 2x HC-SR04 ultrasonic sensors (spaced 20 cm apart, angled slightly inward at 7.5°)
* 4x MAX7219 8x8 LED modules (cascaded into an 8x32 matrix)
* An 8x4 physical measurement grid on the desk (32 cells, 5x5 cm each)

### Why Analytic Trigonometry Fails Here
In theory, a 2D position can be derived from two distance readings via basic triangulation. In practice, HC-SR04 sensors have a beam cone of roughly 15° and produce inconsistent reflections depending on the angle of incidence. A flat target angled toward one sensor reflects the acoustic pulse away from the second sensor. To get reliable stereo readings, a cylindrical target (a highlighter) was used so that both sensors always hit a perpendicular surface normal.

Furthermore, ultrasonic waves propagate spherically. To account for the curved wavefront, the outer columns (Column 1 and Column 4) were shifted slightly toward the sensors, creating an arc-shaped row geometry. Using a 5-sample median filter to reject hardware glitches, all 32 grid cells were calibrated empirically:

| Row | Column 1 (L / R cm) | Column 2 (L / R cm) | Column 3 (L / R cm) | Column 4 (L / R cm) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 29.9 / 24.2 | 29.1 / 29.4 | 29.1 / 29.7 | 25.4 / 29.7 |
| **2** | 33.3 / 29.7 | 33.8 / 34.1 | 33.0 / 34.6 | 31.2 / 36.2 |
| **3** | 37.8 / 36.6 | 38.0 / 37.8 | 37.5 / 38.0 | 37.2 / 37.5 |
| **4** | 42.6 / 40.1 | 43.4 / 42.9 | 42.1 / 43.1 | 40.3 / 42.8 |
| **5** | 46.1 / 45.1 | 47.5 / 47.0 | 47.0 / 48.3 | 45.2 / 47.3 |
| **6** | 50.3 / 49.8 | 51.9 / 51.9 | 51.7 / 51.9 | 50.2 / 50.7 |
| **7** | 55.8 / 55.1 | 57.1 / 56.3 | 56.4 / 56.8 | 54.6 / 55.8 |
| **8** | 60.0 / 58.9 | 62.2 / 61.4 | 61.4 / 62.0 | 59.4 / 60.4 |

### Sensor Resolution Limits in the Center Columns
Looking at the calibration table, the readings for the two center columns (Column 2 and Column 3) are nearly identical. In Row 6, the difference between (6,2) and (6,3) is only 2 millimeters. Because the natural variance of the HC-SR04 exceeds this margin, separating these two columns deterministically is physically impossible with a 20 cm baseline. This hardware limitation is handled directly inside the SNN training rule via spatial tolerance.

---

## System Architecture: Teacher vs. Student

The 8x32 LED matrix is split into two 8x16 sections, where each of the 32 grid cells maps to a 2x2 LED block:
1. **Lower/Right Half (X = 0 to 15) – The Teacher (Ground Truth):** A deterministic Nearest-Neighbor lookup compares live sensor readings against the 32 reference points (Euclidean distance) and displays the closest matching cell.
2. **Upper/Left Half (X = 16 to 31) – The SNN:** The 128 excitatory neurons (32 populations of 4 neurons each) are wired 1:1 to the 128 individual LEDs.

### Asynchronous Execution (Multithreading)
Acquiring 5 ultrasonic samples per sensor takes around 150 to 200 ms due to the speed of sound and required settle times between pings. However, a dynamical spiking network must update on a millisecond timescale so membrane potentials and synaptic traces do not freeze during I/O waits.

To solve this, sensor polling runs asynchronously in a dedicated background thread (`sensor_worker`), while the SNN simulation loop runs freely at 100 Hz (10 ms per step) in the main thread. A second background thread listens for the `Enter` key, ensuring synaptic weight updates only happen when the object has been placed in a cell and hands are clear of the sensor cones.

---

## SNN Dynamics and Mathematical Model

### 1. Temporal Phase Coding
The network operates on a 50-step oscillation cycle (500 ms total). Within the first 35 steps (`sensory_window`), the measured distances from the left and right sensors are mapped linearly to discrete spike arrival times $t_L$ and $t_R$:

$$t_{L,R} = \text{clip}\left( \left\lfloor \frac{d_{L,R} - 22.0}{42.0} \cdot 34 \right\rfloor, 0, 34 \right)$$

Close objects trigger early spikes in the cycle; distant objects trigger late spikes. During the final 15 steps (150 ms) of each cycle, sensory input is shut off completely. For a population to stay lit on the LED matrix during this window, it must sustain its own activity through recurrent self-excitation.

### 2. ALIF Neuron Model (Adaptive Leaky Integrate-and-Fire)
Each of the 32 grid cells is represented by an excitatory population of 4 neurons ($V_{\text{exc}}$) that excite each other via a local synaptic trace ($\text{trace}_{\text{exc}}$). All 32 populations also project into a shared pool of 8 inhibitory neurons ($V_{\text{inh}}$), which feeds back a global inhibitory trace ($\text{trace}_{\text{inh}}$) to suppress competing populations.

At every time step $t$, state variables decay exponentially:

$$V_{\text{exc}}(t) = \alpha_v \cdot V_{\text{exc}}(t-1)$$

$$\text{trace}_{\text{exc}}(t) = \alpha_{\text{trace}} \cdot \text{trace}_{\text{exc}}(t-1)$$

$$a(t) = \alpha_{\text{adapt}} \cdot a(t-1)$$

The membrane potential is then updated using the incoming sensory trace ($\text{trace}_{\text{in}}$), recurrent self-excitation, global inhibition, the spike-frequency adaptation variable $a(t)$, and Gaussian membrane noise $\eta(t)$:

$$V_{\text{exc}}(t) \leftarrow V_{\text{exc}}(t) + \text{trace}_{\text{in}}(t) + w_{\text{exc,exc}} \cdot \text{trace}_{\text{exc}}(t) - w_{\text{inh,exc}} \cdot \text{trace}_{\text{inh}}(t) - a(t) + \eta(t)$$

When a neuron crosses the firing threshold ($\theta = 1.0$), it emits a spike ($S_{\text{exc}} = 1$), its membrane potential resets to $-0.15$, and its corresponding LED is triggered.

### 3. Stabilizing the Attractor Dynamics
Two dynamical instabilities had to be solved during development:
* **Multistability (Runaway Excitation):** Once two different grid cells were trained, strong recurrent weights (`w_exc_exc`) caused both populations to stay active simultaneously even after the object moved. Introducing spike-frequency adaptation $a(t)$ (which increments by $\beta_{\text{adapt}} = 0.02$ on every spike) gradually fatigues active neurons, allowing the network to drop old states at the cycle boundary.
* **Self-Shutoff via Synchronous Inhibition:** When all 8 inhibitory neurons shared the exact same firing threshold of $1.0$, the first burst of a winning population caused all 8 inhibitory neurons to fire simultaneously. The resulting inhibitory pulse was strong enough to wipe out the winning population itself, causing the LEDs to flash only once per 500 ms cycle. By assigning graded thresholds (`np.linspace(0.50, 1.35, 8)`) to the inhibitory pool, inhibition scales smoothly with total network activity: a single active population recruits just enough inhibition to silence background noise while continuing to pulse multiple times per cycle, whereas two competing populations recruit the full inhibitory pool until the weaker one drops out.

### 4. Online Error-Driven Learning (LTP / Distance-Weighted LTD)
Input weights $W_L$ and $W_R$ are initialized randomly. When `Enter` is pressed, the network evaluates the total spike counts across all 32 populations at the end of the 500 ms cycle:
* **Center-Column Tolerance:** If the teacher locates the object in Column 2, spiking activity in Column 3 of the same row is not penalized (and vice versa), preventing destructive weight oscillations caused by sensor noise.
* **Targeted Depression (LTD):** Any non-target population $p$ outside the tolerance zone that fired at least one spike has its input weights around the active spike timing ($t_L \pm 1, t_R \pm 1$) reduced. The penalty scales with both the number of stray spikes and the physical Euclidean distance ($d_{\text{grid}}$) between the misfiring cell and the target cell:
  $$\Delta W_{\text{LTD}} = 0.03 + 0.015 \cdot \min(\text{Spikes}_p, 6) + 0.01 \cdot d_{\text{grid}}$$
* **Potentiation (LTP):** If the target population fired fewer than 12 spikes during the cycle, or fired less than a competing population, its input weights at ($t_L \pm 1, t_R \pm 1$) are increased by $+0.12$. Weights are clipped to `[0.02, 0.45]` so that a single sensor spike can never push a neuron over the $1.0$ threshold without coincidence from the second sensor.