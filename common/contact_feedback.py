"""仅从 MuJoCo 接触约束解读取工具受力，不用几何穿透量构造力。"""
import mujoco
import numpy as np


def contact_force(model, data, tool_id, surface_id, normal_world):
    """返回工具所受世界系合力、法向投影和每个接触点的原始数据。"""
    normal = np.asarray(normal_world, dtype=float)
    if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("接触面法向必须是三维单位向量")
    total = np.zeros(3)  # 没有匹配接触时保持精确的零向量。
    points = []
    for index in range(data.ncon):  # ncon 是有效接触数，不遍历预分配数组的无效部分。
        contact = data.contact[index]
        g1, g2 = int(contact.geom[0]), int(contact.geom[1])
        if {g1, g2} != {tool_id, surface_id}:
            continue  # 不计入夹爪、连接杆、连杆或地面之间的其他接触。
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)  # 前三项为力 N，后三项为力矩 N·m。
        frame = contact.frame.reshape(3, 3)  # 每一行是一个接触坐标轴在世界系的方向。
        sign = 1.0 if g2 == tool_id else -1.0  # 接触法向从 geom1 指向 geom2；输出统一为工具受力。
        world = sign * (frame.T @ local[:3])
        total += world  # 多个匹配接触点的世界系力逐个相加。
        points.append({
            "time_s": float(data.time), "contact_index": index, "geom1_id": g1, "geom2_id": g2,
            "constraint_address": int(contact.efc_address), "contact_distance_m": float(contact.dist),
            "tool_force_sign": sign,
            **{f"contact_pos_{axis}_m": float(contact.pos[i]) for i, axis in enumerate("xyz")},
            **{f"frame_{i}{j}": float(frame[i, j]) for i in range(3) for j in range(3)},
            **{f"local_force_{axis}_n": float(local[i]) for i, axis in enumerate(("normal", "tangent1", "tangent2"))},
            **{f"local_torque_{i}_nm": float(local[i + 3]) for i in range(3)},
            **{f"tool_force_{axis}_n": float(world[i]) for i, axis in enumerate("xyz")},
        })
    return {"force_world_n": total, "normal_force_n": float(normal @ total),
            "tool_contact_count": len(points), "points": points}
