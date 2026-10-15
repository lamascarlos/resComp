"""Reservoir computing with AgNW networks.

This module wraps the AgNW simulation as a reservoir computer.
The network's internal dynamics (stochastic junction switching) provide
high-dimensional, nonlinear state evolution with memory.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.linalg import lstsq

from .geometry import build_network, Network
from .circuit import solve_circuit, total_current, junction_currents
from .dualnode import DualNodeCircuit
from .cached_solver import CachedSolver
from .dynamics import SimParams, update_dynamics, update_dynamics_lifetime
from .simulate import SimulationResult


@dataclass
class ReservoirState:
    """Reservoir state at one time step."""
    t: float
    V_in: float
    I_total: float
    G_junctions: np.ndarray       # all junction conductances (n_junctions,)
    V_nodes: np.ndarray           # all node voltages (n_wires+2,)
    sampled_G: Optional[np.ndarray] = None    # sampled junction conductances (n_sampled,)
    sampled_ids: Optional[np.ndarray] = None


@dataclass
class ReservoirResult:
    """Result of a reservoir run."""
    network: Network
    states: list[ReservoirState]
    params: SimParams
    wire_diameter: Optional[float] = None
    input_signal: Optional[np.ndarray] = None   # the input signal used (for reference)


class InputEncoder:
    """Encode input signals into voltage waveforms for the AgNW network."""
    
    def __init__(
        self,
        n_inputs: int = 1,
        encoding: str = "amplitude",      # "amplitude", "frequency", "phase", "spatial", "time_mux"
        base_frequency: float = 1.0,
        amplitude_scale: float = 1.0,
        electrode_pairs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            n_inputs: Number of input channels
            encoding: Encoding scheme:
                - "amplitude": input modulates amplitude of sine wave
                - "frequency": input modulates frequency
                - "phase": input modulates phase
                - "spatial": different inputs applied to different electrode pairs
                - "time_mux": inputs time-multiplexed sequentially
            base_frequency: Base frequency (Hz) for sinusoidal carrier
            amplitude_scale: Scaling factor for amplitude modulation
            electrode_pairs: For spatial encoding, list of (electrode_in, electrode_gnd) wire indices
        """
        self.n_inputs = n_inputs
        self.encoding = encoding
        self.base_frequency = base_frequency
        self.amplitude_scale = amplitude_scale
        self.electrode_pairs = electrode_pairs
        
        if encoding == "spatial" and electrode_pairs is None:
            raise ValueError("spatial encoding requires electrode_pairs")
        if encoding == "spatial" and len(electrode_pairs) < n_inputs:
            raise ValueError(f"need at least {n_inputs} electrode pairs for spatial encoding")
    
    def encode(self, inputs: np.ndarray, t: np.ndarray, dt: float) -> list[Callable[[float], float]]:
        """
        Generate voltage waveform functions for each input channel.
        
        Args:
            inputs: (n_steps, n_inputs) or (n_steps,) input signal
            t: Time array (n_steps,)
            dt: Time step
            
        Returns:
            List of waveform functions waveform_fn(t) -> V_in for each input channel.
            For single-electrode operation, returns single function combining all inputs.
        """
        inputs = np.asarray(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        n_steps, n_in = inputs.shape
        assert n_in == self.n_inputs
        
        if self.encoding == "amplitude":
            # Amplitude modulation: V(t) = sum_i input_i(t) * sin(2*pi*f*t)
            # Each input gets its own frequency for separation
            def make_waveform(i):
                freq = self.base_frequency * (i + 1)
                def wf(t_val):
                    # Interpolate input at time t_val
                    idx = int(t_val / dt)
                    if idx >= n_steps:
                        idx = n_steps - 1
                    return self.amplitude_scale * inputs[idx, i] * np.sin(2 * np.pi * freq * t_val)
                return wf
            return [make_waveform(i) for i in range(self.n_inputs)]
        
        elif self.encoding == "frequency":
            # Frequency modulation: V(t) = sin(2*pi*(f0 + k*input)*t)
            def make_waveform(i):
                def wf(t_val):
                    idx = int(t_val / dt)
                    if idx >= n_steps:
                        idx = n_steps - 1
                    freq = self.base_frequency + self.amplitude_scale * inputs[idx, i]
                    return np.sin(2 * np.pi * freq * t_val)
                return wf
            return [make_waveform(i) for i in range(self.n_inputs)]
        
        elif self.encoding == "phase":
            # Phase modulation: V(t) = sin(2*pi*f*t + k*input)
            def make_waveform(i):
                def wf(t_val):
                    idx = int(t_val / dt)
                    if idx >= n_steps:
                        idx = n_steps - 1
                    phase = self.amplitude_scale * inputs[idx, i]
                    return np.sin(2 * np.pi * self.base_frequency * t_val + phase)
                return wf
            return [make_waveform(i) for i in range(self.n_inputs)]
        
        elif self.encoding == "time_mux":
            # Time multiplexing: cycle through inputs
            steps_per_input = n_steps // self.n_inputs
            def combined_waveform(t_val):
                idx = int(t_val / dt)
                if idx >= n_steps:
                    idx = n_steps - 1
                input_idx = (idx // steps_per_input) % self.n_inputs
                return self.amplitude_scale * inputs[idx, input_idx]
            return [combined_waveform]
        
        elif self.encoding == "spatial":
            # Spatial encoding: each input on different electrode pair
            # Returns one waveform per electrode pair
            def make_waveform(i):
                def wf(t_val):
                    idx = int(t_val / dt)
                    if idx >= n_steps:
                        idx = n_steps - 1
                    return self.amplitude_scale * inputs[idx, i]
                return wf
            return [make_waveform(i) for i in range(self.n_inputs)]
        
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")


class AgNWReservoir:
    """AgNW network as a reservoir computer."""
    
    def __init__(
        self,
        n_wires: int,
        area: float,
        length: float,
        params: SimParams,
        rng: Optional[np.random.Generator] = None,
        wire_diameter: Optional[float] = None,
        solver: str = "direct",
        n_sampled_junctions: int = 200,
        cell_size: Optional[float] = None,
        band_frac: float = 0.05,
        network: Optional[Network] = None,
        network_seed: Optional[int] = None,
    ):
        """
        Args:
            n_wires: Number of nanowires
            area: Substrate area (m^2)
            length: Wire length (m)
            params: Simulation parameters (switching dynamics)
            rng: Random number generator
            wire_diameter: Wire diameter for dual-node model (None for wire-as-node)
            solver: "direct" or "cached"
            n_sampled_junctions: Number of junctions to record
            cell_size: Spatial hashing cell size
            band_frac: Electrode band fraction
            network: Pre-built network (optional)
            network_seed: Seed for network generation (if network not provided)
        """
        self.n_wires = n_wires
        self.area = area
        self.length = length
        self.params = params
        self.wire_diameter = wire_diameter
        self.solver = solver
        self.n_sampled_junctions = n_sampled_junctions
        self.cell_size = cell_size
        self.band_frac = band_frac
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        
        # Build or use provided network
        if network is not None:
            self.network = network
            if network.n_wires != n_wires:
                raise ValueError(f"network has {network.n_wires} wires but n_wires={n_wires}")
        else:
            net_rng = np.random.default_rng(network_seed) if network_seed is not None else self.rng
            self.network = build_network(n_wires, area, length, net_rng, cell_size, band_frac)
        
        self.n_junctions = self.network.junctions.shape[0]
        
        # Setup circuit
        self.circuit = None if wire_diameter is None else DualNodeCircuit(self.network, wire_diameter)
        self.cached = CachedSolver(self.circuit) if solver == "cached" else None
        
        # Junction sampling for state readout
        if self.n_sampled_junctions > 0 and self.n_junctions > 0:
            self.sampled_junction_ids = self.rng.choice(
                self.n_junctions, size=min(self.n_sampled_junctions, self.n_junctions), replace=False
            )
        else:
            self.sampled_junction_ids = np.empty(0, dtype=int)
        
        # State
        self.G = np.full(self.n_junctions, params.G_off)
        self.age = np.zeros(self.n_junctions, dtype=int)
        self._step = 0
        self._dt = params.dt
    
    def reset(self, rng: Optional[np.random.Generator] = None):
        """Reset reservoir to initial state (all junctions OFF)."""
        if rng is not None:
            self.rng = rng
        self.G = np.full(self.n_junctions, self.params.G_off)
        self.age = np.zeros(self.n_junctions, dtype=int)
        self._step = 0
    
    def step(self, V_in: float) -> ReservoirState:
        """Run one simulation step with input voltage V_in."""
        dt = self.params.dt
        
        # Solve circuit
        if self.circuit is None:
            V = solve_circuit(self.network, self.G, V_in)
            I_total = total_current(self.network, self.G, V)
        else:
            V = self.circuit.solve(self.G, V_in) if self.cached is None else self.cached.solve(self.G, V_in)
            I_total = self.circuit.total_current(V)
        
        # Compute junction voltages and currents
        if self.n_junctions > 0:
            if self.circuit is None:
                k = self.network.junctions[:, 0]
                l = self.network.junctions[:, 1]
                dV = V[k] - V[l]
                I_mem = junction_currents(self.network, self.G, V)
            else:
                dV = self.circuit.junction_voltage_drops(V)
                I_mem = self.G * dV
            V_mem = np.abs(dV)
            
            # Update junction dynamics
            if self.params.lifetime_steps:
                self.G, self.age = update_dynamics_lifetime(
                    self.G, self.age, V_mem, I_mem, self.params, self.rng
                )
            else:
                self.G = update_dynamics(self.G, V_mem, I_mem, self.params, self.rng)
        
        # Record state
        state = ReservoirState(
            t=self._step * dt,
            V_in=V_in,
            I_total=I_total,
            G_junctions=self.G.copy(),
            V_nodes=V.copy(),
            sampled_G=self.G[self.sampled_junction_ids].copy() if len(self.sampled_junction_ids) > 0 else None,
            sampled_ids=self.sampled_junction_ids.copy() if len(self.sampled_junction_ids) > 0 else None,
        )
        
        self._step += 1
        return state
    
    def run(
        self,
        input_signal: np.ndarray,
        input_encoder: InputEncoder,
        record_every: int = 1,
        warmup_steps: int = 0,
    ) -> ReservoirResult:
        """
        Run reservoir with input signal.
        
        Args:
            input_signal: (n_steps, n_inputs) or (n_steps,) input time series
            input_encoder: InputEncoder instance
            record_every: Record state every N steps
            warmup_steps: Number of steps to run with zero input before processing signal
            
        Returns:
            ReservoirResult with recorded states
        """
        input_signal = np.asarray(input_signal)
        if input_signal.ndim == 1:
            input_signal = input_signal[:, None]
        n_steps, n_inputs = input_signal.shape
        
        if n_inputs != input_encoder.n_inputs:
            raise ValueError(f"input_signal has {n_inputs} channels but encoder expects {input_encoder.n_inputs}")
        
        dt = self.params.dt
        t = np.arange(n_steps + warmup_steps) * dt
        
        # Generate waveform functions
        waveforms = input_encoder.encode(input_signal, t[warmup_steps:], dt)
        
        states = []
        
        # Warmup
        for _ in range(warmup_steps):
            self.step(0.0)
        
        # Process input
        for i in range(n_steps):
            # Combine waveforms (for single-electrode operation, sum them)
            if len(waveforms) == 1:
                V_in = waveforms[0](t[warmup_steps + i])
            else:
                # Multiple electrode pairs: would need multi-electrode circuit support
                # For now, sum them (superposition)
                V_in = sum(wf(t[warmup_steps + i]) for wf in waveforms)
            
            state = self.step(V_in)
            
            if i % record_every == 0:
                states.append(state)
        
        return ReservoirResult(
            network=self.network,
            states=states,
            params=self.params,
            wire_diameter=self.wire_diameter,
            input_signal=input_signal,
        )


class Readout:
    """Linear readout from reservoir states (ridge regression)."""
    
    def __init__(self, ridge_alpha: float = 1e-6):
        """
        Args:
            ridge_alpha: Ridge regularization parameter
        """
        self.ridge_alpha = ridge_alpha
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self._feature_fn: Optional[Callable] = None
    
    def _extract_features(self, states: list[ReservoirState]) -> np.ndarray:
        """Extract feature matrix from reservoir states."""
        if self._feature_fn is not None:
            return np.array([self._feature_fn(s) for s in states])
        
        # Default: use sampled junction conductances + total current
        features = []
        for s in states:
            feats = [s.I_total]
            if s.sampled_G is not None:
                feats.extend(s.sampled_G)
            features.append(feats)
        return np.array(features)
    
    def set_feature_function(self, fn: Callable[[ReservoirState], np.ndarray]):
        """Set custom feature extraction function."""
        self._feature_fn = fn
    
    def train(
        self,
        states: list[ReservoirState],
        targets: np.ndarray,
        feature_fn: Optional[Callable[[ReservoirState], np.ndarray]] = None,
    ):
        """
        Train readout weights using ridge regression.
        
        Args:
            states: List of ReservoirState
            targets: (n_samples, n_outputs) target values
            feature_fn: Optional custom feature extraction
        """
        if feature_fn is not None:
            X = np.array([feature_fn(s) for s in states])
        else:
            X = self._extract_features(states)
        
        y = np.asarray(targets)
        if y.ndim == 1:
            y = y[:, None]
        
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        # Add bias column
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        
        # Ridge regression: (X^T X + alpha*I) w = X^T y
        # Solve for each output independently
        self.weights = np.zeros((n_features + 1, n_outputs))
        
        # Use scipy's lstsq with regularization
        for i in range(n_outputs):
            # Regularized normal equations
            A = X_bias.T @ X_bias + self.ridge_alpha * np.eye(n_features + 1)
            b = X_bias.T @ y[:, i]
            w, *_ = lstsq(A, b)
            self.weights[:, i] = w
    
    def predict(self, states: list[ReservoirState]) -> np.ndarray:
        """Predict outputs for given states."""
        if self.weights is None:
            raise ValueError("Readout not trained. Call train() first.")
        
        X = self._extract_features(states)
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_bias @ self.weights
    
    def predict_one(self, state: ReservoirState) -> np.ndarray:
        """Predict output for a single state."""
        if self.weights is None:
            raise ValueError("Readout not trained. Call train() first.")
        
        if self._feature_fn is not None:
            x = self._feature_fn(state)
        else:
            feats = [state.I_total]
            if state.sampled_G is not None:
                feats.extend(state.sampled_G)
            x = np.array(feats)
        
        x_bias = np.append(x, 1.0)
        return x_bias @ self.weights


def default_feature_fn(state: ReservoirState) -> np.ndarray:
    """Default feature extraction: total current + sampled junction conductances."""
    feats = [state.I_total]
    if state.sampled_G is not None:
        feats.extend(state.sampled_G)
    return np.array(feats)


def conductance_feature_fn(state: ReservoirState) -> np.ndarray:
    """Feature extraction: all junction conductances (high-dimensional)."""
    return state.G_junctions.copy()


def voltage_feature_fn(state: ReservoirState) -> np.ndarray:
    """Feature extraction: node voltages (excluding electrode nodes)."""
    return state.V_nodes[:-2].copy()  # exclude V_IN and GND nodes


def combined_feature_fn(state: ReservoirState) -> np.ndarray:
    """Combined features: current + conductances + voltages."""
    feats = [state.I_total]
    if state.sampled_G is not None:
        feats.extend(state.sampled_G)
    feats.extend(state.V_nodes[:-2])
    return np.array(feats)