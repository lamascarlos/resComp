#!/usr/bin/env python3
"""Time one integration step of an N10 pad network of a given side at a given drive voltage, after 200 warm-up steps
(filaments formed), and report the peak memory. Used to size runs on new machines (2026-10-08, cluster test).

    python scripts/bench_step.py <side_mm> <voltage_V>
"""
import sys, time, resource, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from agnw_rc import config as C
from agnw_rc.electrodes import make_layout
from agnw_rc.multicircuit import MultiTerminalCircuit
from agnw_rc.volatile import conductance, junction_kd0, step_state
side = float(sys.argv[1]); v = float(sys.argv[2])
t0 = time.time()
net = C.make_network(1, side=side); els = make_layout("pads_scaled", net)
_, wd = C.network_params(); mc = MultiTerminalCircuit(net, wd, els)
vp = C.volatile_params(None, tau_s=0.3, tau_spread=1.0, dt=2e-3)
kd0 = junction_kd0(vp, len(net.junctions), 1); st = np.zeros((2, len(net.junctions)))
fixed = mc.fixed_electrodes; driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
Vf = np.zeros(len(fixed)); Vf[driven] = v
t1 = time.time()
for i in range(200):        # warm-up: let filaments form at the operating voltage
    V = mc.solve(conductance(st, vp), Vf, method="pcg"); st = step_state(st, mc.junction_voltage_drops(V), vp, kd0)
t2 = time.time()
for i in range(50):
    V = mc.solve(conductance(st, vp), Vf, method="pcg"); st = step_state(st, mc.junction_voltage_drops(V), vp, kd0)
t3 = time.time()
print(f"side {side}: wires {net.wire_endpoints.shape[0]}, junctions {len(net.junctions)}, build {t1-t0:.0f} s, "
      f"step {(t3-t2)/50*1e3:.0f} ms (formed), peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
