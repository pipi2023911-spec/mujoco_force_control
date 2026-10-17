"""一维导纳 M*a+D*v+K*x=F_ext；x正方向与外力一致，不使用目标力误差。"""
import numpy as np


class Admittance:
    def __init__(self, config):
        self.mass = config["admittance_mass_kg"]
        self.damping = config["admittance_damping_ns_m"]
        self.stiffness = config["admittance_stiffness_n_m"]
        self.limit = config["admittance_offset_limit_m"]
        self.speed = config["admittance_speed_limit_m_s"]
        values = [self.mass, self.damping, self.stiffness, self.limit, self.speed]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("虚拟质量、阻尼、刚度和限幅必须是有限正数")
        self.position = 0.0
        self.velocity = 0.0

    def step(self, external_force, dt):
        if not np.isfinite([external_force, dt]).all() or dt <= 0:
            raise ValueError("导纳输入必须有限，周期必须为正")
        # 孤立线性虚拟系统的半隐式Euler稳定条件；接触闭环仍须实际验证。
        if self.stiffness * dt**2 / self.mass + 2 * self.damping * dt / self.mass >= 4:
            raise ValueError("当前周期不满足虚拟系统半隐式积分稳定条件")
        old_position, old_velocity = self.position, self.velocity
        acceleration = (external_force - self.damping * old_velocity - self.stiffness * old_position) / self.mass
        raw_velocity = old_velocity + acceleration * dt
        velocity = float(np.clip(raw_velocity, -self.speed, self.speed))
        raw_position = old_position + velocity * dt
        position = float(np.clip(raw_position, -self.limit, self.limit))
        limited = abs(raw_velocity - velocity) > 1e-15 or abs(raw_position - position) > 1e-15
        # 饱和时同步内部速度与实际位移，避免积累不可执行的状态。
        self.velocity = (position - old_position) / dt
        self.position = position
        return position, {"admittance_x_m": position, "admittance_v_m_s": self.velocity,
                          "admittance_a_m_s2": acceleration, "admittance_limited": bool(limited)}
