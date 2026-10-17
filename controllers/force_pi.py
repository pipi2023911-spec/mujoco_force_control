"""一维位置型力 PI；仅接收数值，不依赖 MuJoCo 或 ROS。"""
import numpy as np


class ForcePI:
    def __init__(self, config):
        self.kp = config["pi_kp_m_per_n"]  # m/N；输出为向平面压入的参考修正。
        self.ki = config["pi_ki_m_per_ns"]  # m/(N·s)；integral 保存积分贡献，单位 m。
        self.integral_limit = config["pi_integral_limit_m"]
        self.offset_limit = config["pi_offset_limit_m"]
        self.speed = config["pi_offset_speed_m_s"]
        values = [self.kp, self.ki, self.integral_limit, self.offset_limit, self.speed]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("PI 增益及限幅必须为有限正数")
        self.integral = 0.0
        self.offset = 0.0

    def reset(self, error):
        # 接触建立后无突跳接管：积分项抵消初始比例项，初始修正为零。
        self.integral = float(np.clip(-self.kp * error, -self.integral_limit, self.integral_limit))
        self.offset = 0.0

    def step(self, target_force, measured_force, dt):
        if not np.isfinite([target_force, measured_force, dt]).all() or dt <= 0:
            raise ValueError("PI 输入必须有限，周期必须大于零")
        error = float(target_force - measured_force)
        proposed = float(np.clip(self.integral + self.ki * error * dt,
                                 -self.integral_limit, self.integral_limit))
        lower = max(-self.offset_limit, self.offset - self.speed * dt)
        upper = min(self.offset_limit, self.offset + self.speed * dt)
        raw = self.kp * error + proposed
        applied = float(np.clip(raw, lower, upper))
        blocked = (raw - applied) * error > 0
        if blocked:
            # 幅值/变化率饱和且误差继续推动饱和时，暂停积分累积。
            raw = self.kp * error + self.integral
            applied = float(np.clip(raw, lower, upper))
        else:
            self.integral = proposed
        self.offset = applied
        return applied, {"force_error_n": error, "pi_p_m": self.kp * error,
                         "pi_i_m": self.integral, "pi_offset_m": applied,
                         "pi_unsaturated_m": raw, "pi_integral_blocked": bool(blocked),
                         "pi_output_limited": bool(abs(raw - applied) > 1e-15)}
