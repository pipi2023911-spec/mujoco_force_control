"""三语言值零阶 Sugeno 在线增益调度；复用固定 PI 的积分与输出限制。"""
import numpy as np

from controllers.force_pi import ForcePI


class FuzzyAdaptivePI(ForcePI):
    # 行：误差 N/Z/P；列：误差变化率 N/Z/P。表项为基准增益倍率。
    KP_RULES = np.array([[1.3, 1.1, .8], [.9, 1., .9], [.8, 1.1, 1.3]])
    KI_RULES = np.array([[1.2, 1.1, .65], [.7, 1.1, .7], [.65, 1.1, 1.2]])

    def __init__(self, config):
        super().__init__(config)
        self.base_kp, self.base_ki = self.kp, self.ki
        self.error_scale = config.get("fuzzy_error_scale_n", 1.)
        self.rate_scale = config.get("fuzzy_rate_scale_n_s", 5.)
        self.derivative_tau = config.get("fuzzy_derivative_tau_s", .05)
        self.gain_tau = config.get("fuzzy_gain_tau_s", .1)
        self.kp_bounds = config.get("fuzzy_kp_bounds", [self.kp * .5, self.kp * 1.5])
        self.ki_bounds = config.get("fuzzy_ki_bounds", [self.ki * .5, self.ki * 1.5])
        values = [self.error_scale, self.rate_scale, self.derivative_tau, self.gain_tau,
                  *self.kp_bounds, *self.ki_bounds]
        if (len(self.kp_bounds) != 2 or len(self.ki_bounds) != 2 or
                not np.isfinite(values).all() or min(values) <= 0 or
                not self.kp_bounds[0] <= self.kp <= self.kp_bounds[1] or
                not self.ki_bounds[0] <= self.ki <= self.ki_bounds[1]):
            raise ValueError("模糊归一化、时间常数必须有限正数，增益区间必须包含基准值")
        self.previous_error = None
        self.filtered_rate = 0.

    @staticmethod
    def memberships(value):
        """N/Z/P 线性肩形/三角隶属函数，和恒为 1。"""
        x = float(np.clip(value, -1., 1.))
        return np.array([max(-x, 0.), 1. - abs(x), max(x, 0.)])

    def reset(self, error):
        if not np.isfinite(error):
            raise ValueError("模糊 PI 初始误差必须有限")
        self.kp, self.ki = self.base_kp, self.base_ki
        super().reset(error)
        self.previous_error = float(error)
        self.filtered_rate = 0.

    def step(self, target_force, measured_force, dt):
        if not np.isfinite([target_force, measured_force, dt]).all() or dt <= 0:
            raise ValueError("模糊 PI 输入必须有限，周期必须为正")
        error = float(target_force - measured_force)
        raw_rate = 0. if self.previous_error is None else (error - self.previous_error) / dt
        self.filtered_rate += dt / (self.derivative_tau + dt) * (raw_rate - self.filtered_rate)
        weights = np.outer(self.memberships(error / self.error_scale),
                           self.memberships(self.filtered_rate / self.rate_scale))
        kp_target = float(np.clip(self.base_kp * np.sum(weights * self.KP_RULES), *self.kp_bounds))
        ki_target = float(np.clip(self.base_ki * np.sum(weights * self.KI_RULES), *self.ki_bounds))
        alpha = dt / (self.gain_tau + dt)
        old_kp = self.kp
        self.kp += alpha * (kp_target - self.kp)
        self.ki += alpha * (ki_target - self.ki)
        # integral 已经是 m，不因 Ki 变化整体缩放；补偿 Kp 更新的瞬时比例跳变。
        compensated = self.integral + (old_kp - self.kp) * error
        self.integral = float(np.clip(compensated, -self.integral_limit, self.integral_limit))
        self.previous_error = error
        offset, details = super().step(target_force, measured_force, dt)
        details.update(pi_kp_m_per_n=self.kp, pi_ki_m_per_ns=self.ki,
                       fuzzy_error_normalized=float(np.clip(error / self.error_scale, -1, 1)),
                       fuzzy_rate_normalized=float(np.clip(self.filtered_rate / self.rate_scale, -1, 1)),
                       fuzzy_filtered_rate_n_s=self.filtered_rate,
                       fuzzy_kp_target=kp_target, fuzzy_ki_target=ki_target,
                       fuzzy_compensation_limited=bool(self.integral_limit < abs(compensated)))
        return offset, details
