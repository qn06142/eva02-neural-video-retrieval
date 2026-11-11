"""Cooperative CUDA pacing for laptop training, including validation/recompute."""
import math
import time
import subprocess

import torch


def read_gpu_temperature():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        timeout=3, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return float(result.stdout.strip().splitlines()[0])


class GpuPacer:
    def __init__(self, duty_cycle=1., min_pause_ms=0., target_temp=None, temperature_reader=None):
        if not math.isfinite(duty_cycle) or not 0 < duty_cycle <= 1:
            raise ValueError("GPU duty cycle must lie in (0, 1]")
        if not math.isfinite(min_pause_ms) or min_pause_ms < 0:
            raise ValueError("GPU minimum pause must be finite and nonnegative")
        self.duty_cycle = duty_cycle
        self.minimum = min_pause_ms / 1000
        self.rested = 0.
        if target_temp is not None and (not math.isfinite(target_temp) or not 30 <= target_temp <= 90):
            raise ValueError("GPU target temperature must be between 30 and 90 C")
        self.target_temp = target_temp
        self.maximum = duty_cycle
        self.temperature_reader = temperature_reader or read_gpu_temperature
        self.last_check = float('-inf')
        self.last_log = float('-inf')

    def refresh_temperature(self):
        now = time.perf_counter()
        if self.target_temp is None or now - self.last_check < 5:
            return
        self.last_check = now
        previous = self.duty_cycle
        try:
            temperature = self.temperature_reader()
            if not math.isfinite(temperature):
                raise ValueError('Invalid GPU temperature')
            if temperature >= self.target_temp + 5:
                print(f'[gpu_pacing] cooling pause: {temperature:.1f}C; '
                      f'waiting for <= {self.target_temp-2:.1f}C before more CUDA work', flush=True)
                while temperature > self.target_temp - 2:
                    before = time.perf_counter()
                    time.sleep(5)
                    self.rested += time.perf_counter() - before
                    temperature = self.temperature_reader()
                    if not math.isfinite(temperature):
                        raise ValueError('Invalid GPU temperature')
                self.duty_cycle = min(self.duty_cycle, .3)
                self.last_check = time.perf_counter()
                print(f'[gpu_pacing] cooling pause ended at {temperature:.1f}C', flush=True)
            if temperature >= self.target_temp + 1:
                self.duty_cycle = max(min(.15, self.maximum), self.duty_cycle - .05)
            elif temperature < self.target_temp - 2:
                self.duty_cycle = min(self.maximum, self.duty_cycle + .025)
            if self.duty_cycle != previous or now - self.last_log >= 60:
                print(f'[gpu_pacing] temperature={temperature:.1f}C target={self.target_temp:.1f}C '
                      f'duty={self.duty_cycle:.3f} ceiling={self.maximum:.3f}', flush=True)
                self.last_log = now
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.duty_cycle = min(self.duty_cycle, .3)
            if now - self.last_log >= 60:
                print(f'[gpu_pacing] temperature unavailable; conservative duty={self.duty_cycle:.3f}: {exc}', flush=True)
                self.last_log = now

    def call(self, function, *args, device=None, **kwargs):
        if device is None:
            device = next((value.device for value in (*args, *kwargs.values())
                           if isinstance(value, torch.Tensor)), None)
        device = torch.device(device) if device is not None else None
        if device is None or device.type != "cuda":
            return function(*args, **kwargs)
        self.refresh_temperature()
        if self.duty_cycle == 1 and not self.minimum:
            return function(*args, **kwargs)
        torch.cuda.synchronize(device)
        start, previous_rest = time.perf_counter(), self.rested
        try:
            return function(*args, **kwargs)
        finally:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            nested_rest = self.rested - previous_rest
            work = max(0., elapsed - nested_rest)
            # Inner encoder calls may already have rested during recomputation.
            # Count that rest rather than sleeping for the same work twice.
            pause = max(0., self.minimum - nested_rest,
                        work * (1/self.duty_cycle - 1) - nested_rest)
            if pause:
                before = time.perf_counter()
                time.sleep(pause)
                self.rested += time.perf_counter() - before


_pacer = GpuPacer()


def configure_gpu_pacing(duty_cycle=.3, min_pause_ms=50., target_temp=None):
    global _pacer
    _pacer = GpuPacer(duty_cycle, min_pause_ms, target_temp)


def paced_gpu_call(function, *args, device=None, **kwargs):
    return _pacer.call(function, *args, device=device, **kwargs)
