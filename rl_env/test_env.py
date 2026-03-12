from marl_uav.env.core import World
from marl_uav.env.scenarios.uav_mission import Scenario

def test():
    scenario = Scenario()
    world = scenario.make_world()

    print(f"无人机数量：{len(world.agents)}")
    print(f"无人机质量：{world.agents[0].mass} {world.agents[1].mass} {world.agents[2].mass} {world.agents[3].mass}")
    world.step()

if __name__ == "__main__":
    test()