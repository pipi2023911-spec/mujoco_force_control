"""两个实验共用的可视窗口：先观察、空格开始、结束后保留末帧。"""
import copy
import math
import time

import mujoco


class ExperimentView:
    """显示副本与实验数据隔离，鼠标拖拽不会改变验收实验。"""

    def __init__(self, model, enabled, speed=1.0):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("播放速度必须大于零")
        self.enabled = enabled
        self.speed = speed
        self.started = False
        self.handle = None
        if enabled:
            self.model = copy.copy(model)  # 窗口内修改物理选项也不影响原实验模型。
            self.data = mujoco.MjData(self.model)

    def __enter__(self):
        if self.enabled:
            import mujoco.viewer
            self.handle = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.on_key)
        return self

    def on_key(self, key):
        if key == 32:  # GLFW 的空格键编码；只用于开始，不改变物理控制命令。
            self.started = True

    def display(self, source):
        if not self.enabled:
            return
        self.data.qpos[:] = source.qpos
        self.data.qvel[:] = source.qvel
        self.data.ctrl[:] = source.ctrl
        self.data.time = source.time
        mujoco.mj_forward(self.model, self.data)  # 刷新显示副本的几何位置。
        self.handle.sync()

    def wait_start(self, source):
        if self.enabled:
            print("初始场景已就绪：点击 MuJoCo 窗口，按空格开始；可先旋转、缩放观察。", flush=True)
            while not self.started:
                if not self.handle.is_running():
                    raise RuntimeError("实验尚未开始，窗口已关闭")
                self.display(source)
                time.sleep(0.02)
        self.wall_start = time.perf_counter()  # 从开始键之后计时，排除准备时间。

    def update(self, source):
        if self.enabled:
            if not self.handle.is_running():
                raise RuntimeError("实验尚未完成，窗口已关闭")
            self.display(source)
            time.sleep(max(0, self.wall_start + source.time / self.speed - time.perf_counter()))

    def wait_end(self, source):
        if self.enabled:
            print("物理过程已结束：窗口保留最终状态，可继续观察。关闭窗口后生成验收报告。", flush=True)
            while self.handle.is_running():
                self.display(source)  # 只刷新显示，不再执行 mj_step，不增加实验时长。
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()
