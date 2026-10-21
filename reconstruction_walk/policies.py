"""One command/action contract, with policy-owned Isaac plant configuration."""


class LocomotionController:
    def __init__(self, name, robot_cfg, device="cpu"):
        import torch
        self.torch, self.name, self.device = torch, name, device
        if name == "wbt":
            from spark_policy.control.whole_body.unitree_g1.wbt import UnitreeG1BatchedWBTPolicy
            self.policy = UnitreeG1BatchedWBTPolicy(robot_cfg, num_envs=1, device=device)
        elif name == "sport":
            from spark_policy.control.whole_body.unitree_g1.sport import UnitreeG1SportPolicy
            self.policy = UnitreeG1SportPolicy(robot_cfg, num_envs=1, device=device)
        else:
            raise ValueError(f"Unknown walking policy: {name}")
        self.steps = 0
        self.blend = 0.
        self.idle_steps = 0

    @staticmethod
    def plant_config(name):
        if name == "wbt":
            from spark_policy.control.whole_body.unitree_g1.wbt import wbt_isaac_actuation_config
            return wbt_isaac_actuation_config()
        if name == "sport":
            return {}  # Existing Sport backend defaults; do not apply WBT drives.
        raise ValueError(name)

    def reset(self):
        self.policy.reset()
        self.steps = 0
        self.blend = 0.
        self.idle_steps = 0

    def infer(self, feedback, command, upper_body):
        torch = self.torch
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device).reshape(1, 3)
        common = dict(body_joint_pos=feedback["body_joint_pos"], body_joint_vel=feedback["body_joint_vel"],
                      root_quat_xyzw=feedback["root_pose_w"][:, 3:7],
                      root_angular_velocity=feedback["root_angular_velocity_b"],
                      upper_body_target=upper_body)
        if self.steps < 20:
            cmd = torch.zeros_like(cmd)
        if self.name == "sport":
            target, info = self.policy.infer_tensor(**common, velocity_command=cmd)
        else:
            moving = bool(torch.any(cmd.abs() > .005).item())
            self.idle_steps = 0 if moving else self.idle_steps + 1
            velocity = feedback.get("root_velocity_w")
            speed = float(torch.linalg.vector_norm(velocity[:, :2]).item()) if velocity is not None else 0.
            desired = 1. if moving else (0. if self.idle_steps >= 15 and speed < .12 else self.blend)
            self.blend += max(-.02, min(.02, desired - self.blend))
            if self.blend <= 0:
                target, info = self.policy.infer_tensor(**common, command=None, mode="squat", stance=False)
            elif self.blend >= 1:
                target, info = self.policy.infer_tensor(**common, command=cmd, mode="loco", stance=False)
            else:
                squat, si = self.policy.infer_tensor(**common, command=None, mode="squat", stance=False)
                loco, li = self.policy.infer_tensor(**common, command=cmd, mode="loco", stance=False)
                target = (1 - self.blend) * squat + self.blend * loco
                info = {k: (1 - self.blend) * si[k] + self.blend * li[k] for k in ("motor_kps", "motor_kds")}
        self.steps += 1
        action_info = {"target_actuated_pos": target}
        action_info.update({k: info[k] for k in ("motor_kps", "motor_kds") if k in info})
        return target, action_info
