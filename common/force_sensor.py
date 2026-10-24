"""法向力测量模型：延迟 → 偏置和白噪声 → 一阶低通；参数不是硬件规格。"""
from collections import deque
import math
import numbers

import numpy as np


class ForceSensor:
    def __init__(self, config, control_dt):
        self.noise_std = float(config.get("sensor_noise_std_n", 0.0))
        self.bias = float(config.get("sensor_bias_n", 0.0))
        delay = config.get("sensor_delay_steps", 0)
        cutoff = config.get("sensor_lpf_cutoff_hz")
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("传感器采样周期必须为有限正数")
        if not math.isfinite(self.noise_std) or self.noise_std < 0 or not math.isfinite(self.bias):
            raise ValueError("噪声标准差必须有限非负，偏置必须有限")
        if isinstance(delay, bool) or not isinstance(delay, numbers.Integral) or delay < 0:
            raise ValueError("传感器延迟步数必须为非负整数")
        self.delay_steps = int(delay)
        self.control_dt = control_dt
        if cutoff is None:
            self.alpha = 1.0
        else:
            if not math.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("低通截止频率必须为有限正数，或用 null 关闭")
            tau = 1.0 / (2.0 * math.pi * cutoff)
            self.alpha = control_dt / (control_dt + tau)
        seed = config.get("sensor_seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError("传感器随机种子必须为非负整数")
        self.rng = np.random.default_rng(seed)
        self.buffer = deque(maxlen=self.delay_steps + 1)
        self.filtered = None
        self.sample_index = 0

    def step(self, true_force, time_s=None):
        """每个控制周期调用一次；启动时延迟队列不足，保持最早的真实样本。"""
        if time_s is None:
            time_s = self.sample_index * self.control_dt
        if not math.isfinite(true_force) or not math.isfinite(time_s):
            raise ValueError("力真值和采样时间必须有限")
        self.buffer.append((float(true_force), float(time_s)))
        delayed, source_time = self.buffer[0]
        raw = delayed if self.bias == 0 else delayed + self.bias
        if self.noise_std > 0:
            raw += float(self.rng.normal(0.0, self.noise_std))
        # 无低通时直接赋值，避免 x += 1*(u-x) 的消减误差破坏默认直通。
        if self.filtered is None or self.alpha == 1.0:
            self.filtered = raw
        else:
            self.filtered += self.alpha * (raw - self.filtered)
        self.sample_index += 1
        return self.filtered, {
            "sensor_true_force_n": float(true_force),
            "sensor_delayed_force_n": delayed,
            "sensor_raw_force_n": raw,
            "sensor_filtered_force_n": self.filtered,
            "sensor_sample_time_s": float(time_s),
            "sensor_source_time_s": source_time,
            "sensor_actual_delay_s": float(time_s) - source_time,
            "sensor_delay_ready": len(self.buffer) == self.delay_steps + 1,
        }
