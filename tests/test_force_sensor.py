"""纯数值测量通道测试；可由 unittest 或 pytest 执行。"""
import math
import unittest

import numpy as np

from common.force_sensor import ForceSensor


class ForceSensorTests(unittest.TestCase):
    def test_default_exact_passthrough(self):
        sensor = ForceSensor({}, .005)
        # 5 -> .01 可揭示旧补丁的消减误差；不是近似比较。
        for force in [0., 5., .01, -3., 1e-12, 1e6, .1]:
            value, info = sensor.step(force)
            self.assertEqual(value, force)
            self.assertEqual(info["sensor_actual_delay_s"], 0.)
        self.assertTrue(np.signbit(sensor.step(-0.)[0]))

    def test_delay_source_time_and_startup(self):
        sensor = ForceSensor({"sensor_delay_steps": 2}, .005)
        results = [sensor.step(x, i * .005) for i, x in enumerate([1., 2., 3., 4.])]
        self.assertEqual([x[0] for x in results], [1., 1., 1., 2.])
        self.assertEqual([x[1]["sensor_delay_ready"] for x in results], [False, False, True, True])
        self.assertAlmostEqual(results[-1][1]["sensor_actual_delay_s"], .01)
        self.assertEqual(results[-1][1]["sensor_source_time_s"], .005)

    def test_bias_and_no_filter(self):
        sensor = ForceSensor({"sensor_bias_n": -.2}, .005)
        self.assertEqual(sensor.step(5.)[0], 4.8)
        self.assertEqual(sensor.step(0.)[0], -.2)  # 不将负测量值截成零。

    def test_lowpass_step_response(self):
        sensor = ForceSensor({"sensor_lpf_cutoff_hz": 10.}, .005)
        sensor.step(0.)
        alpha = .005 / (.005 + 1 / (2 * math.pi * 10))
        for i in range(1, 21):
            value, _ = sensor.step(5.)
            self.assertAlmostEqual(value, 5 * (1 - (1 - alpha)**i))

    def test_noise_seed_and_statistics(self):
        config = {"sensor_noise_std_n": .2, "sensor_seed": 7}
        a, b = ForceSensor(config, .005), ForceSensor(config, .005)
        values = [a.step(5.)[0] for _ in range(20000)]
        self.assertEqual(values, [b.step(5.)[0] for _ in values])
        self.assertLess(abs(np.mean(values) - 5.), .005)
        self.assertLess(abs(np.std(values) - .2), .005)
        other = ForceSensor({**config, "sensor_seed": 8}, .005)
        self.assertNotEqual(values[:10], [other.step(5.)[0] for _ in range(10)])

    def test_invalid_config(self):
        cases = [{"sensor_noise_std_n": x} for x in [-1, math.nan, math.inf]]
        cases += [{"sensor_bias_n": math.nan}]
        cases += [{"sensor_delay_steps": x} for x in [-1, 1.5, True]]
        cases += [{"sensor_lpf_cutoff_hz": x} for x in [0, -1, math.nan, math.inf]]
        cases += [{"sensor_seed": x} for x in [-1, 1.5, True]]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                ForceSensor(config, .005)
        for dt in [0, -1, math.nan, math.inf]:
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                ForceSensor({}, dt)

    def test_invalid_sample(self):
        sensor = ForceSensor({}, .005)
        for force, time in [(math.nan, 0), (math.inf, 0), (0, math.nan)]:
            with self.assertRaises(ValueError):
                sensor.step(force, time)


if __name__ == "__main__":
    unittest.main()
