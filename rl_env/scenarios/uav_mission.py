import numpy as np
from marl_uav.env.core import World,Agent,Landmark
from marl_uav.env.config import DRONE_CONFIGS,MAP_SIZE,NUM_OBSTACLES,OBSTACLE_RADIUS,MIN_GOAL_DIST,GOAL_RADIUS,POS_NOISE_STD,VEL_NOISE_STD,OBS_CONFIG,NUM_OBSTACLES
from marl_uav.env.cover_scan import GridMapScan


class Scenario:
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.dim_p = 2
        formation = ['SCOUT','SCOUT','RELAY','EXECUTOR']

        world.landmarks=[]
        for i in range (NUM_OBSTACLES):
            landmark = Landmark()
            landmark.name = f'obstacle_{i}'
            landmark.collide = True
            landmark.movable = False
            landmark.size = OBSTACLE_RADIUS
            landmark.state.p_vel = np.zeros(world.dim_p)
            world.landmarks.append(landmark)

        world.agents = []
        for i,uav_type in enumerate(formation):
            agent = Agent(uav_type=uav_type)
            agent.name = f'uva_{i}_{uav_type}'
            agent.collide = True
            agent.silent = False

            agent.state.p_pos = np.random.uniform(-MAP_SIZE/2,MAP_SIZE/2,world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.layer = 1.0
            agent.state.update_height()
            world.agents.append(agent)

        world.goal_pos = np.array([40.0,40.0])
        self.scanner = GridMapScan(map_size=MAP_SIZE)
        return world

    def reset_world(self,world):
        self.scanner.reset()
        world.goal_pos = np.random.uniform(-MAP_SIZE/2+5,MAP_SIZE/2-5,world.dim_p)
        while True:
            start_center = np.random.uniform(-MAP_SIZE/2+10,MAP_SIZE/2-10,world.dim_p)
            if np.linalg.norm(start_center-world.goal_pos)>MIN_GOAL_DIST:
                break
        for i,agent in enumerate(world.agents):
            agent.lambda_collision = 5.0
            agent.lambda_lr = 0.05
            while True:
                noise = np.random.uniform(-3, 3, world.dim_p)
                proposed_pos = start_center+noise
                collision = False
                for obs in world.landmarks:
                    if np.linalg.norm(proposed_pos-obs.state.p_pos)<=(obs.size+agent.size+0.5):
                        collision = True
                        break
                for other in world.agents[:i]:
                    if np.linalg.norm(proposed_pos-other.state.p_pos)<=(other.size+agent.size+0.5):
                        collision = True
                        break
                if not collision:
                    agent.state.p_pos = proposed_pos
                    break
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.accumulated_fatigue = 0.0
            agent.is_weak_battery = False
            agent.state.layer = np.random.uniform(0.5,2.0)
            agent.state.update_height()
            agent.last_action_u = np.zeros(world.dim_p)
        for landmark in world.landmarks:
            while True:
                pos = np.random.uniform(-MAP_SIZE/2,MAP_SIZE/2,world.dim_p)
                if np.linalg.norm(pos-start_center)>10.0 and np.linalg.norm(pos-world.goal_pos)>10.0:
                    landmark.state.p_pos = pos
                    break

    def reward(self,world,agent):
        reward = 0
        reward -= 0.01
        r=agent.get_sensing_radius()
        new_cells = self.scanner.update_coverage(agent.state.p_pos,r)
        reward += new_cells*1.0
        dist_to_goal = np.linalg.norm(agent.state.p_pos-world.goal_pos)
        reward -= 0.1*(dist_to_goal/MAP_SIZE)
        if agent.uav_type == 'SCOUT':
            reward += 0.05*new_cells
        elif agent.uav_type == 'EXECUTOR':
            reward -= 0.05*(dist_to_goal/MAP_SIZE)
        if dist_to_goal < GOAL_RADIUS:
            reward += 5.0
        reward *= agent.reward_weight

        alignment_reward = 0.0
        n_neighbors = 0
        comm_consistency_reward = 0.0
        for other in world.agents:
            if other is not agent:
                distance = np.linalg.norm(other.state.p_pos-agent.state.p_pos)
                if distance<agent.max_comm:
                    dot_prod = np.dot(agent.state.p_vel,other.state.p_vel)
                    alignment_reward += dot_prod
                    link_quality = 1.0-(distance/agent.max_comm)
                    comm_consistency_reward += link_quality
                    n_neighbors += 1
        if n_neighbors>0:
            reward += 0.05*(alignment_reward/n_neighbors)
            reward += 0.01*n_neighbors
            reward += 0.05*(comm_consistency_reward/n_neighbors)
            if agent.uav_type == 'RELAY':
                reward += 0.02*n_neighbors

        if agent.is_weak_battery:
            action_mag = np.linalg.norm(agent.action.u)
            penalty = 0.01*action_mag/agent.battery_capacity
            reward -= penalty

        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                distance_2d = np.linalg.norm(a.state.p_pos - agent.state.p_pos)
                distance_z = abs(a.state.height-agent.state.height)
                min_distance_2d = agent.size+a.size
                min_distance_z = 2.0
                safety_margin = 2.0
                if distance_2d<min_distance_2d + safety_margin and distance_z<min_distance_z:
                    penalty_field = 1.0/(distance_2d-min_distance_2d+0.1)
                    penalty_field = np.clip(penalty_field,0.0,20.0)
                    reward -= penalty_field*0.1
                if distance_2d < min_distance_2d and distance_z < min_distance_z:
                    violation = (min_distance_2d-distance_2d)+(min_distance_z-distance_z)
                    reward -= agent.lambda_collision*violation*5.0
                    agent.lambda_collision += agent.lambda_lr*violation
                    if not hasattr(world,'collisions'):world.collisions=0
                    world.collisions += 1
                else:
                    agent.lambda_collision = max(1.0,agent.lambda_collision-agent.lambda_lr*0.05)
            for obs in world.landmarks:
                distance = np.linalg.norm(obs.state.p_pos-agent.state.p_pos)
                collision_distance = obs.size+agent.size
                safe_margin_obs = 3.0
                if distance < collision_distance+safe_margin_obs:
                    distance_to_surface = distance-collision_distance
                    if distance_to_surface<0:
                        distance_to_surface = 0.001
                    repulsion = 1.0/distance_to_surface
                    repulsion = np.clip(repulsion,0.0,20.0)
                    reward -= repulsion*0.2
                if distance < collision_distance:
                    violation = collision_distance-distance
                    reward -= agent.lambda_collision*violation*5.0
                    agent.lambda_collision += agent.lambda_lr*violation
                    if not hasattr(world,'collisions'):world.collisions=0
                    world.collisions += 1

        jerk_cost = 0.0
        if agent.last_action_u is not None:
            jerk = np.linalg.norm(agent.action.u-agent.last_action_u)
            reward -= jerk*0.3
            jerk_cost = jerk if agent.last_action_u is not None else 0.0
        agent.last_action_u = agent.action.u
        energy_cost = np.sum(np.square(agent.action.u))
        return reward,{
            "energy":energy_cost,
            "jerk":jerk_cost,
            "lambda_collision":agent.lambda_collision
        }

    def get_comm(self, world):
        adj = np.zeros((len(world.agents), len(world.agents)))
        for i,agent_a in enumerate(world.agents):
            for j,agent_b in enumerate(world.agents):
                if i == j:
                    continue
                distance = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
                if distance < agent_a.max_comm:
                    adj[i][j] = 1
        return adj

    def observation(self,agent,world):
        vel_noise = np.random.normal(0,VEL_NOISE_STD,world.dim_p)
        pos_noise = np.random.normal(0,POS_NOISE_STD,world.dim_p)
        measured_pos = agent.state.p_pos+pos_noise
        measured_vel = agent.state.p_vel+vel_noise
        goal_rel = world.goal_pos-measured_pos
        last_u = agent.last_action_u if agent.last_action_u is not None else np.zeros(2)
        self.state = [
            measured_vel[0]/agent.max_speed,
            measured_vel[1]/agent.max_speed,
            measured_pos[0]/(MAP_SIZE/2),
            measured_pos[1]/(MAP_SIZE/2),
            1.0 if agent.is_weak_battery else 0.0,
            agent.state.height/20.0,
            goal_rel[0]/MAP_SIZE,
            goal_rel[1]/MAP_SIZE,
            last_u[0],
            last_u[1],
        ]

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos-measured_pos)
        entity_obs = np.concatenate(entity_pos) if entity_pos else np.array([])
        other_obs = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            distance = np.linalg.norm(other_agent.state.p_pos-measured_pos)
            is_visible = (distance<=agent.max_comm)
            if is_visible:
                rel_pos = other_agent.state.p_pos-measured_pos
                other_obs.extend([rel_pos[0],rel_pos[1],1.0])
            else:
                other_obs.extend([0.0,0.0,0.0])
        final_obs = np.concatenate([
            np.array(self.state),
            entity_obs,
            np.array(other_obs),
        ])
        expected_dim =(
            OBS_CONFIG['dim_self']+
            NUM_OBSTACLES*OBS_CONFIG['dim_obs_item']+
            (len(world.agents)-1)*OBS_CONFIG['dim_neigh_item']
        )
        assert len(final_obs)==expected_dim
        return final_obs
