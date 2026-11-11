from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scrape"))

import torch

from gpu_pacing import GpuPacer


class FakeClock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def tick(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class PacingTests(unittest.TestCase):
    def test_temperature_feedback_and_ceiling(self):
        clock = FakeClock()
        reading = [73.]
        pacer = GpuPacer(.6, 10, 70, lambda: reading[0])
        with patch('gpu_pacing.time.perf_counter', clock.tick), patch('builtins.print'):
            pacer.refresh_temperature()
            self.assertAlmostEqual(pacer.duty_cycle, .55)
            reading[0] = 60.
            clock.now += 6
            pacer.refresh_temperature()
            self.assertAlmostEqual(pacer.duty_cycle, .575)
            for _ in range(10):
                clock.now += 6
                pacer.refresh_temperature()
            self.assertAlmostEqual(pacer.duty_cycle, .6)
            reading[0] = 70.
            clock.now += 6
            pacer.refresh_temperature()
            self.assertAlmostEqual(pacer.duty_cycle, .6)

    def test_hot_gpu_waits_for_hysteresis_before_work(self):
        clock = FakeClock()
        readings = iter([80., 74., 68.])
        pacer = GpuPacer(.6, 10, 70, lambda: next(readings))
        with patch('gpu_pacing.time.perf_counter', clock.tick), \
             patch('gpu_pacing.time.sleep', clock.sleep), patch('builtins.print'):
            pacer.refresh_temperature()
        self.assertEqual(clock.sleeps, [5, 5])
        self.assertAlmostEqual(pacer.rested, 10.)
        self.assertLessEqual(pacer.duty_cycle, .3)

    def test_temperature_failure_backs_off_and_invalid_target_rejected(self):
        def unavailable():
            raise OSError('test')
        pacer = GpuPacer(.6, 10, 70, unavailable)
        with patch('builtins.print'):
            pacer.refresh_temperature()
        self.assertAlmostEqual(pacer.duty_cycle, .3)
        with self.assertRaises(ValueError):
            GpuPacer(.6, 10, float('nan'))

    def test_work_receives_rest_and_preserves_result(self):
        clock = FakeClock()
        pacer = GpuPacer(.25, 50)
        def work():
            clock.now += .1
            return 7
        with patch('gpu_pacing.torch.cuda.synchronize') as sync, \
             patch('gpu_pacing.time.perf_counter', clock.tick), \
             patch('gpu_pacing.time.sleep', clock.sleep):
            self.assertEqual(pacer.call(work, device='cuda'), 7)
        self.assertAlmostEqual(sum(clock.sleeps), .3)
        self.assertEqual(sync.call_count, 2)

    def test_nested_calls_do_not_double_count_rest(self):
        clock = FakeClock()
        pacer = GpuPacer(.25, 50)
        def inner():
            clock.now += .1
        def outer():
            pacer.call(inner, device='cuda')
        with patch('gpu_pacing.torch.cuda.synchronize'), \
             patch('gpu_pacing.time.perf_counter', clock.tick), \
             patch('gpu_pacing.time.sleep', clock.sleep):
            pacer.call(outer, device='cuda')
        self.assertAlmostEqual(sum(clock.sleeps), .3)

    def test_exceptions_still_allow_rest(self):
        clock = FakeClock()
        pacer = GpuPacer(.5, 50)
        def work():
            clock.now += .01
            raise ValueError('test')
        with patch('gpu_pacing.torch.cuda.synchronize'), \
             patch('gpu_pacing.time.perf_counter', clock.tick), \
             patch('gpu_pacing.time.sleep', clock.sleep):
            with self.assertRaises(ValueError):
                pacer.call(work, device='cuda')
        self.assertAlmostEqual(sum(clock.sleeps), .05)

    def test_cpu_and_disabled_mode_do_not_touch_cuda(self):
        with patch('gpu_pacing.torch.cuda.synchronize') as sync, patch('gpu_pacing.time.sleep') as sleep:
            self.assertEqual(GpuPacer(.3, 50).call(lambda x: x+1, torch.tensor(2)).item(), 3)
            self.assertEqual(GpuPacer(1, 0).call(lambda: 9, device='cuda'), 9)
            sync.assert_not_called()
            sleep.assert_not_called()

    def test_invalid_settings(self):
        for duty, pause in ((0, 0), (2, 0), (float('nan'), 0), (.3, -1)):
            with self.assertRaises(ValueError):
                GpuPacer(duty, pause)


if __name__ == '__main__':
    unittest.main()
