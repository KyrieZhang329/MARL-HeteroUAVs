from __future__ import annotations

import argparse
import json
import random
import sys
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import DefaultDict, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def install_legacy_env_aliases() -> None:
    import rl_env.config as config_module

    marl_uav_module = sys.modules.setdefault("marl_uav", types.ModuleType("marl_uav"))
    env_module = sys.modules.setdefault("marl_uav.env", types.ModuleType("marl_uav.env"))
    env_module.__path__ = []
    marl_uav_module.env = env_module
    sys.modules["marl_uav.env"] = env_module
    sys.modules["marl_uav.env.config"] = config_module
    env_module.config = config_module

    import rl_env.core as core_module

    sys.modules["marl_uav.env.core"] = core_module
    env_module.core = core_module


install_legacy_env_aliases()

from algorithms.mappo import MAPPOAgent
from algorithms.masac import MASACAgent
from rl_env.config import GOAL_RADIUS, MAP_SIZE, MIN_GOAL_DIST, NUM_OBSTACLES, OBSTACLE_RADIUS, OBS_CONFIG, POS_NOISE_STD, VEL_NOISE_STD
from rl_env.core import Agent, Landmark, World


REWARD_COMPONENT_NAMES: Tuple[str, ...] = (
    "step_penalty",
    "coverage_reward",
    "goal_distance_penalty",
    "scout_bonus",
    "executor_distance_penalty",
    "goal_bonus",
    "alignment_bonus",
    "neighbor_bonus",
    "comm_consistency_bonus",
    "relay_bonus",
    "weak_battery_penalty",
    "agent_proximity_penalty",
    "agent_collision_penalty",
    "obstacle_proximity_penalty",
    "obstacle_collision_penalty",
    "jerk_penalty",
)

AUX_METRIC_NAMES: Tuple[str, ...] = (
    "energy_metric",
    "new_cells_metric",
    "distance_to_goal_metric",
    "lambda_collision_metric",
)

FORMATION: Tuple[str, ...] = ("SCOUT", "SCOUT", "RELAY", "EXECUTOR")


def safe_collision_force(self: World, entity_a, entity_b):
    if (not entity_a.collide) or (not entity_b.collide):
        return [None, None]
    if entity_a is entity_b:
        return [None, None]

    delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    if dist < 1e-6:
        delta_pos = np.array([1.0, 0.0], dtype=np.float32)
        dist = 1e-6

    dist_min = entity_a.size + entity_b.size
    penetration = np.logaddexp(0, -(dist - dist_min) / self.contact_margin) * self.contact_margin
    force = self.contact_force * delta_pos / dist * penetration
    force_a = +force if entity_a.movable else None
    force_b = -force if entity_b.movable else None
    return [force_a, force_b]


World.get_collision_force = safe_collision_force


class CoverageScanner:
    def __init__(self, map_size: float = 20.0, grid_num: int = 20):
        self.map_size = map_size
        self.grid_num = grid_num
        self.cell_size = map_size / grid_num
        self.grid = np.zeros((grid_num, grid_num), dtype=np.int8)

    def reset(self) -> None:
        self.grid.fill(0)

    def trans_coordinates(self, pos: np.ndarray) -> Tuple[int, int]:
        offset = self.map_size / 2
        x = np.clip(pos[0], -offset, offset - 0.01)
        y = np.clip(pos[1], -offset, offset - 0.01)
        col = int((x + offset) / self.cell_size)
        row = int((y + offset) / self.cell_size)
        return col, row

    def update_coverage(self, drone_pos: np.ndarray, scan_r: float) -> int:
        r_in_cells = int(scan_r / self.cell_size)
        center_col, center_row = self.trans_coordinates(drone_pos)
        newly_covered_count = 0

        for row in range(center_row - r_in_cells, center_row + r_in_cells + 1):
            for col in range(center_col - r_in_cells, center_col + r_in_cells + 1):
                if 0 <= row < self.grid_num and 0 <= col < self.grid_num:
                    grid_center_x = col * self.cell_size - self.map_size / 2 + self.cell_size / 2
                    grid_center_y = row * self.cell_size - self.map_size / 2 + self.cell_size / 2
                    distance = np.linalg.norm(drone_pos[:2] - np.array([grid_center_x, grid_center_y]))
                    if distance < scan_r and self.grid[row, col] == 0:
                        self.grid[row, col] = 1
                        newly_covered_count += 1
        return newly_covered_count

    def get_coverage_rate(self) -> float:
        return float(np.sum(self.grid) / (self.grid_num ** 2))


@dataclass
class TrainConfig:
    algorithm: str
    total_episodes: int
    max_episode_steps: int
    seed: int
    hidden_size: int
    learning_rate: float
    actor_lr: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    norm_adv: bool
    buffer_size: int
    batch_size: int
    updates_per_step: int
    start_steps: int
    save_every: int
    log_interval: int
    device: str
    run_root: str


class MissionTrainingEnv:
    def __init__(self, max_episode_steps: int, seed: int):
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world: Optional[World] = None
        self.scanner: Optional[CoverageScanner] = None
        self.episode_step = 0
        self.action_dim = 2
        self.obs_dim = (
            OBS_CONFIG["dim_self"]
            + NUM_OBSTACLES * OBS_CONFIG["dim_obs_item"]
            + (len(FORMATION) - 1) * OBS_CONFIG["dim_neigh_item"]
        )

    @property
    def num_agents(self) -> int:
        return len(FORMATION)

    def reset(self) -> np.ndarray:
        self.world = World()
        self.world.dim_c = 2
        self.world.dim_p = 2
        self.world.collisions = 0

        self.world.landmarks = []
        for idx in range(NUM_OBSTACLES):
            landmark = Landmark()
            landmark.name = f"obstacle_{idx}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = OBSTACLE_RADIUS
            landmark.state.p_vel = np.zeros(self.world.dim_p, dtype=np.float32)
            landmark.state.p_pos = np.zeros(self.world.dim_p, dtype=np.float32)
            self.world.landmarks.append(landmark)

        self.world.agents = []
        for idx, uav_type in enumerate(FORMATION):
            agent = Agent(uav_type=uav_type)
            agent.name = f"uav_{idx}_{uav_type}"
            agent.collide = True
            agent.silent = False
            agent.state.p_pos = np.zeros(self.world.dim_p, dtype=np.float32)
            agent.state.p_vel = np.zeros(self.world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(self.world.dim_c, dtype=np.float32)
            agent.action.u = np.zeros(self.action_dim, dtype=np.float32)
            agent.action.c = np.zeros(self.world.dim_c, dtype=np.float32)
            self.world.agents.append(agent)

        self.scanner = CoverageScanner(map_size=MAP_SIZE)
        self.scanner.reset()
        self._reset_world_state()
        self.episode_step = 0
        return self._collect_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
        if self.world is None or self.scanner is None:
            raise RuntimeError("Environment must be reset before stepping.")

        clipped_actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        for idx, agent in enumerate(self.world.agents):
            agent.action.u = clipped_actions[idx].copy()
            agent.action.c = np.zeros(self.world.dim_c, dtype=np.float32)

        self.world.step()
        self.episode_step += 1

        rewards: List[float] = []
        reward_components: DefaultDict[str, float] = defaultdict(float)
        reward_metrics: DefaultDict[str, float] = defaultdict(float)

        for agent in self.world.agents:
            reward, components, metrics = self._compute_reward(agent)
            rewards.append(reward)
            for name in REWARD_COMPONENT_NAMES:
                reward_components[name] += components[name]
            for name in AUX_METRIC_NAMES:
                reward_metrics[name] += metrics[name]

        next_obs = self._collect_observations()
        success = any(np.linalg.norm(agent.state.p_pos - self.world.goal_pos) < GOAL_RADIUS for agent in self.world.agents)
        truncated = self.episode_step >= self.max_episode_steps
        done = success or truncated

        info: Dict[str, object] = {
            "reward_components": dict(reward_components),
            "reward_metrics": dict(reward_metrics),
            "coverage_rate": self.scanner.get_coverage_rate(),
            "collisions": float(getattr(self.world, "collisions", 0)),
            "success": float(success),
            "truncated": float(truncated),
        }
        return next_obs, np.asarray(rewards, dtype=np.float32), done, info

    def _reset_world_state(self) -> None:
        if self.world is None:
            raise RuntimeError("World is not initialized.")

        self.world.goal_pos = self._sample_goal_position()
        start_center = self._sample_start_center(self.world.goal_pos)

        placed_obstacles: List[np.ndarray] = []
        for landmark in self.world.landmarks:
            landmark.state.p_pos = self._sample_obstacle_position(start_center, self.world.goal_pos, placed_obstacles)
            placed_obstacles.append(landmark.state.p_pos.copy())

        for idx, agent in enumerate(self.world.agents):
            agent.lambda_collision = 5.0
            agent.lambda_lr = 0.05
            agent.accumulated_fatigue = 0.0
            agent.is_weak_battery = False
            agent.active = True
            agent.state.p_pos = self._sample_agent_position(start_center, agent, self.world.agents[:idx])
            agent.state.p_vel = np.zeros(self.world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(self.world.dim_c, dtype=np.float32)
            agent.state.layer = float(self.rng.uniform(0.5, 2.0))
            agent.state.update_height()
            agent.action.u = np.zeros(self.action_dim, dtype=np.float32)
            agent.action.c = np.zeros(self.world.dim_c, dtype=np.float32)
            agent.last_action_u = np.zeros(self.action_dim, dtype=np.float32)

    def _sample_goal_position(self) -> np.ndarray:
        low = -MAP_SIZE / 2 + 5
        high = MAP_SIZE / 2 - 5
        return self.rng.uniform(low, high, size=2).astype(np.float32)

    def _sample_start_center(self, goal_pos: np.ndarray) -> np.ndarray:
        low = -MAP_SIZE / 2 + 10
        high = MAP_SIZE / 2 - 10
        for _ in range(1000):
            start_center = self.rng.uniform(low, high, size=2).astype(np.float32)
            if np.linalg.norm(start_center - goal_pos) > MIN_GOAL_DIST:
                return start_center
        raise RuntimeError("Failed to sample a valid start center.")

    def _sample_obstacle_position(
        self,
        start_center: np.ndarray,
        goal_pos: np.ndarray,
        placed_obstacles: List[np.ndarray],
    ) -> np.ndarray:
        low = -MAP_SIZE / 2
        high = MAP_SIZE / 2
        for _ in range(1000):
            pos = self.rng.uniform(low, high, size=2).astype(np.float32)
            if np.linalg.norm(pos - start_center) <= 10.0:
                continue
            if np.linalg.norm(pos - goal_pos) <= 10.0:
                continue
            if any(np.linalg.norm(pos - other_pos) <= (2 * OBSTACLE_RADIUS + 1.0) for other_pos in placed_obstacles):
                continue
            return pos
        raise RuntimeError("Failed to sample a valid obstacle position.")

    def _sample_agent_position(self, start_center: np.ndarray, agent: Agent, existing_agents: List[Agent]) -> np.ndarray:
        if self.world is None:
            raise RuntimeError("World is not initialized.")

        for _ in range(1000):
            noise = self.rng.uniform(-3.0, 3.0, size=self.world.dim_p).astype(np.float32)
            proposed_pos = (start_center + noise).astype(np.float32)
            if np.any(proposed_pos < -MAP_SIZE / 2) or np.any(proposed_pos > MAP_SIZE / 2):
                continue
            if any(np.linalg.norm(proposed_pos - obs.state.p_pos) <= (obs.size + agent.size + 0.5) for obs in self.world.landmarks):
                continue
            if any(np.linalg.norm(proposed_pos - other.state.p_pos) <= (other.size + agent.size + 0.5) for other in existing_agents):
                continue
            return proposed_pos
        raise RuntimeError("Failed to sample a valid agent position.")

    def _collect_observations(self) -> np.ndarray:
        if self.world is None:
            raise RuntimeError("World is not initialized.")
        observations = [self._observe(agent) for agent in self.world.agents]
        return np.stack(observations).astype(np.float32)

    def _observe(self, agent: Agent) -> np.ndarray:
        if self.world is None:
            raise RuntimeError("World is not initialized.")

        vel_noise = self.rng.normal(0.0, VEL_NOISE_STD, size=self.world.dim_p)
        pos_noise = self.rng.normal(0.0, POS_NOISE_STD, size=self.world.dim_p)
        measured_pos = agent.state.p_pos + pos_noise
        measured_vel = agent.state.p_vel + vel_noise
        goal_rel = self.world.goal_pos - measured_pos
        last_u = agent.last_action_u if agent.last_action_u is not None else np.zeros(self.action_dim, dtype=np.float32)

        self_state = np.array(
            [
                measured_vel[0] / agent.max_speed,
                measured_vel[1] / agent.max_speed,
                measured_pos[0] / (MAP_SIZE / 2),
                measured_pos[1] / (MAP_SIZE / 2),
                1.0 if agent.is_weak_battery else 0.0,
                agent.state.height / 20.0,
                goal_rel[0] / MAP_SIZE,
                goal_rel[1] / MAP_SIZE,
                last_u[0],
                last_u[1],
            ],
            dtype=np.float32,
        )

        entity_obs = np.concatenate([entity.state.p_pos - measured_pos for entity in self.world.landmarks]).astype(np.float32)
        other_obs: List[float] = []
        for other_agent in self.world.agents:
            if other_agent is agent:
                continue
            distance = np.linalg.norm(other_agent.state.p_pos - measured_pos)
            if distance <= agent.max_comm:
                rel_pos = other_agent.state.p_pos - measured_pos
                other_obs.extend([float(rel_pos[0]), float(rel_pos[1]), 1.0])
            else:
                other_obs.extend([0.0, 0.0, 0.0])

        observation = np.concatenate([self_state, entity_obs, np.asarray(other_obs, dtype=np.float32)]).astype(np.float32)
        if observation.shape[0] != self.obs_dim:
            raise ValueError(f"Observation dimension mismatch: {observation.shape[0]} != {self.obs_dim}")
        return observation

    def _compute_reward(self, agent: Agent) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        if self.world is None or self.scanner is None:
            raise RuntimeError("World is not initialized.")

        weight = agent.reward_weight
        scan_radius = agent.get_sensing_radius()
        new_cells = self.scanner.update_coverage(agent.state.p_pos, scan_radius)
        dist_to_goal = np.linalg.norm(agent.state.p_pos - self.world.goal_pos)
        action_u = agent.action.u.astype(np.float32)

        components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        metrics = {name: 0.0 for name in AUX_METRIC_NAMES}

        components["step_penalty"] = -0.01 * weight
        components["coverage_reward"] = float(new_cells) * weight
        components["goal_distance_penalty"] = -0.1 * (dist_to_goal / MAP_SIZE) * weight
        if agent.uav_type == "SCOUT":
            components["scout_bonus"] = 0.05 * float(new_cells) * weight
        if agent.uav_type == "EXECUTOR":
            components["executor_distance_penalty"] = -0.05 * (dist_to_goal / MAP_SIZE) * weight
        if dist_to_goal < GOAL_RADIUS:
            components["goal_bonus"] = 5.0 * weight

        alignment_reward = 0.0
        n_neighbors = 0
        comm_consistency_reward = 0.0
        for other in self.world.agents:
            if other is agent:
                continue
            distance = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if distance < agent.max_comm:
                alignment_reward += float(np.dot(agent.state.p_vel, other.state.p_vel))
                comm_consistency_reward += 1.0 - (distance / agent.max_comm)
                n_neighbors += 1
        if n_neighbors > 0:
            components["alignment_bonus"] = 0.05 * (alignment_reward / n_neighbors)
            components["neighbor_bonus"] = 0.01 * n_neighbors
            components["comm_consistency_bonus"] = 0.05 * (comm_consistency_reward / n_neighbors)
            if agent.uav_type == "RELAY":
                components["relay_bonus"] = 0.02 * n_neighbors

        if agent.is_weak_battery:
            action_mag = np.linalg.norm(action_u)
            components["weak_battery_penalty"] = -0.01 * action_mag / agent.battery_capacity

        for other_agent in self.world.agents:
            if other_agent is agent:
                continue
            distance_2d = np.linalg.norm(other_agent.state.p_pos - agent.state.p_pos)
            distance_z = abs(other_agent.state.height - agent.state.height)
            min_distance_2d = agent.size + other_agent.size
            min_distance_z = 2.0
            safety_margin = 2.0
            if distance_2d < min_distance_2d + safety_margin and distance_z < min_distance_z:
                penalty_field = 1.0 / (distance_2d - min_distance_2d + 0.1)
                penalty_field = np.clip(penalty_field, 0.0, 20.0)
                components["agent_proximity_penalty"] -= penalty_field * 0.1
            if distance_2d < min_distance_2d and distance_z < min_distance_z:
                violation = (min_distance_2d - distance_2d) + (min_distance_z - distance_z)
                penalty = agent.lambda_collision * violation * 5.0
                components["agent_collision_penalty"] -= penalty
                agent.lambda_collision += agent.lambda_lr * violation
                self.world.collisions += 1
            else:
                agent.lambda_collision = max(1.0, agent.lambda_collision - agent.lambda_lr * 0.05)

        for obs in self.world.landmarks:
            distance = np.linalg.norm(obs.state.p_pos - agent.state.p_pos)
            collision_distance = obs.size + agent.size
            safe_margin_obs = 3.0
            if distance < collision_distance + safe_margin_obs:
                distance_to_surface = distance - collision_distance
                if distance_to_surface < 0:
                    distance_to_surface = 0.001
                repulsion = 1.0 / distance_to_surface
                repulsion = np.clip(repulsion, 0.0, 20.0)
                components["obstacle_proximity_penalty"] -= repulsion * 0.2
            if distance < collision_distance:
                violation = collision_distance - distance
                penalty = agent.lambda_collision * violation * 5.0
                components["obstacle_collision_penalty"] -= penalty
                agent.lambda_collision += agent.lambda_lr * violation
                self.world.collisions += 1

        if agent.last_action_u is not None:
            jerk = np.linalg.norm(action_u - agent.last_action_u)
            components["jerk_penalty"] -= jerk * 0.3
        metrics["energy_metric"] = float(np.sum(np.square(action_u)))
        agent.last_action_u = action_u.copy()

        metrics["new_cells_metric"] = float(new_cells)
        metrics["distance_to_goal_metric"] = float(dist_to_goal)
        metrics["lambda_collision_metric"] = float(agent.lambda_collision)

        total_reward = float(sum(components.values()))
        return total_reward, components, metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO or SAC agents for the heterogeneous UAV mission environment.")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--total-episodes", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--disable-norm-adv", action="store_false", dest="norm_adv")
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-root", type=str, default="runs")
    parser.set_defaults(norm_adv=True)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_mappo_args(config: TrainConfig) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        norm_adv=config.norm_adv,
        clip_coef=config.clip_coef,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
    )


def build_masac_args(config: TrainConfig) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        actor_lr=config.actor_lr,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
    )


def create_run_directories(config: TrainConfig) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / config.run_root / f"{config.algorithm}_{timestamp}"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, tensorboard_dir, checkpoint_dir


def save_run_config(config: TrainConfig, run_dir: Path, device: torch.device) -> None:
    config_dict = asdict(config)
    config_dict["device"] = str(device)
    (run_dir / "train_config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = next_value.astype(np.float32)

    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
        next_values = values[step]

    returns = advantages + values
    return advantages, returns


def log_episode(
    writer: SummaryWriter,
    episode: int,
    env_steps: int,
    episode_steps: int,
    episode_reward: float,
    total_loss: float,
    value_loss: float,
    policy_loss: float,
    reward_components: Dict[str, float],
    reward_metrics: Dict[str, float],
    coverage_rate: float,
    collisions: float,
    success: float,
) -> None:
    writer.add_scalar("episode/steps", episode_steps, episode)
    writer.add_scalar("episode/total_reward", episode_reward, episode)
    writer.add_scalar("episode/global_agent_steps", env_steps, episode)
    writer.add_scalar("loss/total_loss", total_loss, episode)
    writer.add_scalar("loss/value_loss", value_loss, episode)
    writer.add_scalar("loss/policy_loss", policy_loss, episode)
    writer.add_scalar("env/coverage_rate", coverage_rate, episode)
    writer.add_scalar("env/collisions", collisions, episode)
    writer.add_scalar("env/success", success, episode)

    for name in REWARD_COMPONENT_NAMES:
        writer.add_scalar(f"reward/{name}", reward_components.get(name, 0.0), episode)
    for name in AUX_METRIC_NAMES:
        writer.add_scalar(f"metric/{name}", reward_metrics.get(name, 0.0), episode)


def save_checkpoint(algorithm: str, agent, episode: int, checkpoint_dir: Path) -> None:
    checkpoint_path = checkpoint_dir / f"{algorithm}_episode_{episode:06d}.pt"
    if algorithm == "ppo":
        payload = {
            "episode": episode,
            "algorithm": algorithm,
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
        }
    else:
        payload = {
            "episode": episode,
            "algorithm": algorithm,
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "actor_optimizer": agent.actor_optimizer.state_dict(),
            "q_optimizer": agent.q_optimizer.state_dict(),
            "alpha_optimizer": agent.alpha_optimizer.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
        }
    torch.save(payload, checkpoint_path)


def train_ppo(config: TrainConfig, env: MissionTrainingEnv, writer: SummaryWriter, checkpoint_dir: Path, device: torch.device) -> None:
    agent = MAPPOAgent(env.obs_dim, env.action_dim, env.num_agents, NUM_OBSTACLES, build_mappo_args(config), device)
    env_steps = 0

    for episode in range(1, config.total_episodes + 1):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0.0
        episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        episode_metrics = {name: 0.0 for name in AUX_METRIC_NAMES}
        coverage_rate = 0.0
        collisions = 0.0
        success = 0.0

        rollout_obs: List[np.ndarray] = []
        rollout_actions: List[np.ndarray] = []
        rollout_logprobs: List[np.ndarray] = []
        rollout_values: List[np.ndarray] = []
        rollout_rewards: List[np.ndarray] = []
        rollout_dones: List[float] = []

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions_tensor, logprobs_tensor, _, values_tensor = agent.get_action_values(obs_tensor)

            actions = actions_tensor.cpu().numpy().astype(np.float32)
            next_obs, rewards, done, info = env.step(actions)

            rollout_obs.append(obs.copy())
            rollout_actions.append(actions.copy())
            rollout_logprobs.append(logprobs_tensor.cpu().numpy().astype(np.float32))
            rollout_values.append(values_tensor.squeeze(-1).cpu().numpy().astype(np.float32))
            rollout_rewards.append(rewards.copy())
            rollout_dones.append(float(done))

            episode_steps += 1
            env_steps += env.num_agents
            episode_reward += float(np.sum(rewards))
            for name in REWARD_COMPONENT_NAMES:
                episode_components[name] += float(info["reward_components"][name])
            for name in AUX_METRIC_NAMES:
                episode_metrics[name] += float(info["reward_metrics"][name])
            coverage_rate = float(info["coverage_rate"])
            collisions = float(info["collisions"])
            success = float(info["success"])
            obs = next_obs

        next_value = np.zeros(env.num_agents, dtype=np.float32)
        rewards_array = np.asarray(rollout_rewards, dtype=np.float32)
        values_array = np.asarray(rollout_values, dtype=np.float32)
        dones_array = np.asarray(rollout_dones, dtype=np.float32)
        advantages, returns = compute_gae(rewards_array, values_array, dones_array, next_value, config.gamma, config.gae_lambda)

        batch_obs = torch.as_tensor(np.concatenate(rollout_obs, axis=0), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(np.concatenate(rollout_actions, axis=0), dtype=torch.float32, device=device)
        batch_logprobs = torch.as_tensor(np.concatenate(rollout_logprobs, axis=0), dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        batch_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)

        value_loss, policy_loss = agent.update(batch_obs, batch_actions, batch_logprobs, batch_returns, batch_advantages)
        total_loss = policy_loss + config.vf_coef * value_loss

        log_episode(
            writer=writer,
            episode=episode,
            env_steps=env_steps,
            episode_steps=episode_steps,
            episode_reward=episode_reward,
            total_loss=total_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
            reward_components=episode_components,
            reward_metrics=episode_metrics,
            coverage_rate=coverage_rate,
            collisions=collisions,
            success=success,
        )

        if episode == 1 or episode % config.log_interval == 0:
            print(
                f"[PPO] Episode {episode:05d} | steps={episode_steps:04d} | reward={episode_reward:10.3f} | "
                f"total_loss={total_loss:10.5f} | value_loss={value_loss:10.5f} | policy_loss={policy_loss:10.5f}"
            )

        if episode % config.save_every == 0 or episode == config.total_episodes:
            save_checkpoint("ppo", agent, episode, checkpoint_dir)


def train_sac(config: TrainConfig, env: MissionTrainingEnv, writer: SummaryWriter, checkpoint_dir: Path, device: torch.device) -> None:
    agent = MASACAgent(env.obs_dim, env.action_dim, env.num_agents, NUM_OBSTACLES, build_masac_args(config), device)
    env_steps = 0

    for episode in range(1, config.total_episodes + 1):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0.0
        episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        episode_metrics = {name: 0.0 for name in AUX_METRIC_NAMES}
        coverage_rate = 0.0
        collisions = 0.0
        success = 0.0

        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        total_loss_sum = 0.0
        update_count = 0

        while not done:
            if env_steps < config.start_steps:
                actions = np.random.uniform(-1.0, 1.0, size=(env.num_agents, env.action_dim)).astype(np.float32)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions_tensor, _, _ = agent.actor.get_action(obs_tensor)
                actions = actions_tensor.cpu().numpy().astype(np.float32)

            next_obs, rewards, done, info = env.step(actions)
            for agent_idx in range(env.num_agents):
                agent.buffer.add(obs[agent_idx], actions[agent_idx], rewards[agent_idx], next_obs[agent_idx], float(done))

            if agent.buffer.size >= config.batch_size:
                for _ in range(config.updates_per_step):
                    value_loss, policy_loss = agent.update(config.batch_size)
                    value_loss_sum += float(value_loss)
                    policy_loss_sum += float(policy_loss)
                    total_loss_sum += float(value_loss + policy_loss)
                    update_count += 1

            episode_steps += 1
            env_steps += env.num_agents
            episode_reward += float(np.sum(rewards))
            for name in REWARD_COMPONENT_NAMES:
                episode_components[name] += float(info["reward_components"][name])
            for name in AUX_METRIC_NAMES:
                episode_metrics[name] += float(info["reward_metrics"][name])
            coverage_rate = float(info["coverage_rate"])
            collisions = float(info["collisions"])
            success = float(info["success"])
            obs = next_obs

        avg_value_loss = value_loss_sum / update_count if update_count > 0 else 0.0
        avg_policy_loss = policy_loss_sum / update_count if update_count > 0 else 0.0
        avg_total_loss = total_loss_sum / update_count if update_count > 0 else 0.0

        log_episode(
            writer=writer,
            episode=episode,
            env_steps=env_steps,
            episode_steps=episode_steps,
            episode_reward=episode_reward,
            total_loss=avg_total_loss,
            value_loss=avg_value_loss,
            policy_loss=avg_policy_loss,
            reward_components=episode_components,
            reward_metrics=episode_metrics,
            coverage_rate=coverage_rate,
            collisions=collisions,
            success=success,
        )

        if episode == 1 or episode % config.log_interval == 0:
            print(
                f"[SAC] Episode {episode:05d} | steps={episode_steps:04d} | reward={episode_reward:10.3f} | "
                f"total_loss={avg_total_loss:10.5f} | value_loss={avg_value_loss:10.5f} | policy_loss={avg_policy_loss:10.5f}"
            )

        if episode % config.save_every == 0 or episode == config.total_episodes:
            save_checkpoint("sac", agent, episode, checkpoint_dir)


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)
    set_global_seed(config.seed)

    run_dir, tensorboard_dir, checkpoint_dir = create_run_directories(config)
    save_run_config(config, run_dir, device)

    env = MissionTrainingEnv(max_episode_steps=config.max_episode_steps, seed=config.seed)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    try:
        if config.algorithm == "ppo":
            train_ppo(config, env, writer, checkpoint_dir, device)
        else:
            train_sac(config, env, writer, checkpoint_dir, device)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

