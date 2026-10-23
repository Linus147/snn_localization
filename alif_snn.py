from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import RPi.GPIO as GPIO
import time
import statistics
import math
import numpy as np
import threading

TRIG_L = 24 
ECHO_L = 23
TRIG_R = 17 
ECHO_R = 27 

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG_L, GPIO.OUT)
GPIO.setup(ECHO_L, GPIO.IN)
GPIO.setup(TRIG_R, GPIO.OUT)
GPIO.setup(ECHO_R, GPIO.IN)

def get_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)
    start_time = time.time()
    stop_time = time.time()
    timeout = start_time + 0.06
    while GPIO.input(echo) == 0 and time.time() < timeout:
        start_time = time.time()
    while GPIO.input(echo) == 1 and time.time() < timeout:
        stop_time = time.time()
    if time.time() >= timeout:
        return -1.0
    return ((stop_time - start_time) * 34300) / 2

def get_stable_distance(trig, echo, samples=5):
    readings = []
    for _ in range(samples):
        d = get_distance(trig, echo)
        if 0 < d < 400: 
            readings.append(d)
        time.sleep(0.015)
    if not readings:
        return -1.0
    return statistics.median(readings)

reference_points = [
    (1, 1, 29.9, 24.2), (1, 2, 29.1, 29.4), (1, 3, 29.1, 29.7), (1, 4, 25.4, 29.7),
    (2, 1, 33.3, 29.7), (2, 2, 33.8, 34.1), (2, 3, 33.0, 34.6), (2, 4, 31.2, 36.2),
    (3, 1, 37.8, 36.6), (3, 2, 38.0, 37.8), (3, 3, 37.5, 38.0), (3, 4, 37.2, 37.5),
    (4, 1, 42.6, 40.1), (4, 2, 43.4, 42.9), (4, 3, 42.1, 43.1), (4, 4, 40.3, 42.8),
    (5, 1, 46.1, 45.1), (5, 2, 47.5, 47.0), (5, 3, 47.0, 48.3), (5, 4, 45.2, 47.3),
    (6, 1, 50.3, 49.8), (6, 2, 51.9, 51.9), (6, 3, 51.7, 51.9), (6, 4, 50.2, 50.7),
    (7, 1, 55.8, 55.1), (7, 2, 57.1, 56.3), (7, 3, 56.4, 56.8), (7, 4, 54.6, 55.8),
    (8, 1, 60.0, 58.9), (8, 2, 62.2, 61.4), (8, 3, 61.4, 62.0), (8, 4, 59.4, 60.4)
]

def find_nearest_neighbor(l_meas, r_meas):
    min_dist = float('inf')
    best_match = (1, 1)
    for row, col, l_ref, r_ref in reference_points:
        dist = math.sqrt((l_meas - l_ref)**2 + (r_meas - r_ref)**2)
        if dist < min_dist:
            min_dist = dist
            best_match = (row, col)
    return best_match

current_dist_l = -1.0
current_dist_r = -1.0
target_row = 1
target_col = 1

def sensor_worker():
    global current_dist_l, current_dist_r, target_row, target_col
    while True:
        dl = get_stable_distance(TRIG_L, ECHO_L)
        dr = get_stable_distance(TRIG_R, ECHO_R)
        if dl > 0 and dr > 0:
            current_dist_l = dl
            current_dist_r = dr
            target_row, target_col = find_nearest_neighbor(dl, dr)
        time.sleep(0.04)

threading.Thread(target=sensor_worker, daemon=True).start()

trigger_learning = False
def listen_for_enter():
    global trigger_learning
    while True:
        input()
        trigger_learning = True

threading.Thread(target=listen_for_enter, daemon=True).start()

num_pops = 32
n_per_pop = 4
num_inh = 8
cycle_steps = 50
sensory_window = 35

W_L = np.random.uniform(0.14, 0.22, (sensory_window, num_pops))
W_R = np.random.uniform(0.14, 0.22, (sensory_window, num_pops))

w_exc_exc = 0.145
w_exc_inh = 0.09
w_inh_exc = 0.065

alpha_v = 0.85
alpha_in = 0.80
alpha_trace = 0.84
alpha_adapt = 0.92
beta_adapt = 0.02
threshold = 1.0
thresh_inh = np.linspace(0.50, 1.35, num_inh)

V_exc = np.random.uniform(0.0, 0.1, (num_pops, n_per_pop))
V_inh = np.zeros(num_inh)
trace_in = np.zeros(num_pops)
trace_exc = np.zeros(num_pops)
trace_inh = 0.0
adapt_exc = np.zeros((num_pops, n_per_pop))
led_persistence = np.zeros((num_pops, n_per_pop))
cycle_spike_counts = np.zeros(num_pops)

print("Pulsierendes ALIF-Attraktor-SNN bereit! Drücke ENTER zum Trainieren...")

try:
    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, cascaded=4, block_orientation=-90)
    
    t_step = 0
    t_l = -100
    t_r = -100
    
    while True:
        if t_step == 0:
            if current_dist_l > 0 and current_dist_r > 0:
                t_l = int(np.clip((current_dist_l - 22.0) / 42.0 * (sensory_window - 1), 0, sensory_window - 1))
                t_r = int(np.clip((current_dist_r - 22.0) / 42.0 * (sensory_window - 1), 0, sensory_window - 1))
            else:
                t_l = -100
                t_r = -100
            
            V_exc *= 0.2
            trace_in.fill(0)
            trace_exc *= 0.05
            trace_inh = 0.0
            adapt_exc *= 0.3
            cycle_spike_counts.fill(0)

        V_exc *= alpha_v
        V_inh *= alpha_v
        trace_in *= alpha_in
        trace_exc *= alpha_trace
        trace_inh *= alpha_trace
        adapt_exc *= alpha_adapt
        led_persistence *= 0.65

        V_exc += np.random.normal(0.015, 0.035, (num_pops, n_per_pop))
        V_inh += np.random.normal(0.0, 0.015, num_inh)

        if t_step < sensory_window:
            if t_step == t_l:
                for tau in range(-1, 2):
                    if 0 <= t_l + tau < sensory_window:
                        trace_in += W_L[t_l + tau, :] * (1.0 if tau == 0 else 0.5)
                        
            if t_step == t_r:
                for tau in range(-1, 2):
                    if 0 <= t_r + tau < sensory_window:
                        trace_in += W_R[t_r + tau, :] * (1.0 if tau == 0 else 0.5)

        V_exc += trace_in[:, None]
        V_exc += (trace_exc * w_exc_exc)[:, None]
        V_exc -= trace_inh * w_inh_exc
        V_exc -= adapt_exc

        spikes_exc = V_exc >= threshold
        V_exc[spikes_exc] = -0.15
        adapt_exc[spikes_exc] += beta_adapt
        led_persistence[spikes_exc] = 1.0

        pop_spikes = np.sum(spikes_exc, axis=1)
        trace_exc += pop_spikes
        cycle_spike_counts += pop_spikes

        V_inh += np.sum(pop_spikes) * w_exc_inh
        spikes_inh = V_inh >= thresh_inh
        V_inh[spikes_inh] = -0.20
        trace_inh += np.sum(spikes_inh)

        if t_step == cycle_steps - 1 and trigger_learning and t_l >= 0 and t_r >= 0:
            trigger_learning = False
            target_idx = (target_row - 1) * 4 + (target_col - 1)
            
            allowed_indices = [target_idx]
            if target_col == 2:
                allowed_indices.append((target_row - 1) * 4 + 2)
            elif target_col == 3:
                allowed_indices.append((target_row - 1) * 4 + 1)

            l_slice = slice(max(0, t_l - 1), min(sensory_window, t_l + 2))
            r_slice = slice(max(0, t_r - 1), min(sensory_window, t_r + 2))

            wrong_activity = False
            for p in range(num_pops):
                if p not in allowed_indices and cycle_spike_counts[p] > 3:
                    wrong_activity = True
                    W_L[l_slice, p] -= 0.08
                    W_R[r_slice, p] -= 0.08

            if wrong_activity or cycle_spike_counts[target_idx] < 12:
                W_L[l_slice, target_idx] += 0.12
                W_R[r_slice, target_idx] += 0.12
                
                W_L = np.clip(W_L, 0.02, 0.45)
                W_R = np.clip(W_R, 0.02, 0.45)
                print(f"Lern-Update für Feld ({target_row},{target_col}) | Spikes im Ziel: {int(cycle_spike_counts[target_idx])}")
            else:
                print(f"Feld ({target_row},{target_col}) pulsiert dominant ({int(cycle_spike_counts[target_idx])} Spikes). Kein Update nötig.")

        if t_step % 2 == 0:
            with canvas(device) as draw:
                if current_dist_l > 0 and current_dist_r > 0:
                    t_draw_x = (target_row - 1) * 2
                    t_draw_y = (4 - target_col) * 2
                    draw.rectangle((t_draw_x, t_draw_y, t_draw_x + 1, t_draw_y + 1), fill="white")
                
                active_leds = led_persistence > 0.35
                for p in range(num_pops):
                    if np.any(active_leds[p]):
                        p_row = (p // 4) + 1
                        p_col = (p % 4) + 1
                        base_x = 16 + (p_row - 1) * 2
                        base_y = (4 - p_col) * 2
                        
                        if active_leds[p, 0]:
                            draw.point((base_x, base_y), fill="white")
                        if active_leds[p, 1]:
                            draw.point((base_x + 1, base_y), fill="white")
                        if active_leds[p, 2]:
                            draw.point((base_x, base_y + 1), fill="white")
                        if active_leds[p, 3]:
                            draw.point((base_x + 1, base_y + 1), fill="white")

        t_step = (t_step + 1) % cycle_steps
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nProgramm beendet.")
    GPIO.cleanup()