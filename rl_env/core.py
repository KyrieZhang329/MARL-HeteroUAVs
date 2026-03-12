import numpy as np
from marl_uav.env.config import DRONE_CONFIGS


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        self.layer = 0
        self.height = 0.0

    def update_height(self):
        self.height = self.layer*10.0


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self,uav_type='SCOUT'):
        super().__init__()
        self.uav_type = uav_type
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.setup_heterogeneity(self.uav_type)
        self.accumulated_fatigue = 0.0
        self.is_weak_battery = False
        self.active = True
        self.last_action_u =None

    def setup_heterogeneity(self,uav_type):
        config = DRONE_CONFIGS[uav_type]
        self.color = config['color']
        self.max_speed = config['max_speed']
        self.max_accel = config['max_accel']
        self.fric_coeff = config['fric_coeff']
        self.initial_mass = config['mass']
        self.max_sense = config['max_sense_range']
        self.max_comm = config['max_comm_range']
        self.fov_rad = np.radians(config['fov'])
        self.reward_weight = config['reward_weight']
        self.max_endurance = config['max_endurance']
        self.battery_capacity = config['battery_capacity']
        self.z_cost_factor = config['z_cost_factor']

    def get_sensing_radius(self):
        if not self.active:
            return 0.0
        h = self.state.height
        if h <= 0:
            return 0.0
        geo_radius = h*np.tan(self.fov_rad/2)
        return min(self.max_sense, geo_radius)

    def update_status(self,dt=0.1):
        if not self.active:
            return
        current_speed = np.linalg.norm(self.state.p_vel)
        current_height = self.state.height
        instant_cost = current_speed + (self.z_cost_factor*current_height)
        self.accumulated_fatigue += dt*instant_cost

        if self.accumulated_fatigue > self.max_endurance:
            self.is_weak_battery = True
        else:
            self.is_weak_battery = False


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        sim_steps = 10
        sub_dt = self.dt/sim_steps
        for step in range(sim_steps):
            p_force = [None] * len(self.entities)
            # apply agent physical controls
            p_force = self.apply_action_force(p_force)
            # apply environment forces
            p_force = self.apply_environment_force(p_force)
            # integrate physical state
            self.integrate_state(p_force,dt=sub_dt)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
            if hasattr(agent,'update_status'):
                agent.update_status(self.dt)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                control_input = np.clip(agent.action.u+noise,-1.0,1.0)
                force_action = control_input*agent.max_accel*agent.mass
                force_friction = -np.sign(agent.state.p_vel)*(agent.state.p_vel**2)*agent.fric_coeff
                total_force = force_friction+force_action
                if p_force[i] is None:
                    p_force[i] = total_force
                else:
                    p_force[i] = p_force[i] + force_action
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force,dt):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                            entity.state.p_vel
                            / speed
                            * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
