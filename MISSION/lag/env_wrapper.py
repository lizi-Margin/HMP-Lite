import os, platform, time, gymnasium
import numpy as np
from uhtk.UTIL.colorful import *
from config import GlobalConfig as cfg
from ..example import BaseEnv, get_N_AGENT_EACH_TEAM, get_N_TEAM
from .JSBSim import BVREnv as Env
class MD2D(object):
    @staticmethod
    def get_n_actions(space: gymnasium.Space) -> int:
        if isinstance(space, gymnasium.spaces.Discrete):
            return space.n
        elif isinstance(space, gymnasium.spaces.Box):
            raise NotImplementedError
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
            return np.prod(space.nvec)
        else:
            raise ValueError(f"Unsupported action space type: {type(space)}")
    
    def __init__(self, md: gymnasium.spaces.MultiDiscrete):
        self.md = md
        self.nvec = md.nvec
        self.n = self.get_n_actions(md)
    
    def index_to_action(self, index):
        indices = np.array(np.unravel_index(int(index), self.nvec), dtype=self.md.dtype)
        return indices

# please register this ScenarioConfig into MISSION/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
    '''

    # <Part 1> Needed by the hmp core #
    AGENT_ID_EACH_TEAM = [[0, 1], [2, 3]]
    TEAM_NAMES = ['ALGORITHM.None->None', 'ALGORITHM.None->None',]


    # <Part 2> Needed by env itself #
    MaxEpisodeStep = 400
    render = True
    scenario_name = "2v2/ShootMissile/HierarchySelfplay"
    interested_team = 0

    # <Part 3> Needed by some ALGORITHM #
    _temp_env = Env(scenario_name)  # BUG the scenario_name has not been overridden

    EntityOriented = False
    obs_shape = (_temp_env.task.obs_length,)
    
    n_actions = MD2D.get_n_actions(_temp_env.action_space)
    
    AvailActProvided = True
    StateProvided = True

def make_env(env_name, rank):
    return LAGWrapper(rank)

class LAGWrapper(BaseEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.id = rank
        self.render = ScenarioConfig.render and (self.id==0)
        self.n_teams = get_N_TEAM(ScenarioConfig)
        assert self.n_teams == 2, 'only support 2 teams'
        self.interested_team = ScenarioConfig.interested_team
        assert  self.interested_team in range(0, self.n_teams)
        assert self.interested_team == 0, 'if not so, errors may occur'

        self.n_each_team = get_N_AGENT_EACH_TEAM(ScenarioConfig)
        self.id_each_team = ScenarioConfig.AGENT_ID_EACH_TEAM
        for n_agent in self.n_each_team:
            assert n_agent == self.n_each_team[0], 'all teams must have the same num of agents'
            assert n_agent == 2, 'only 2 agents is supported now'
        self.EntityOriented = ScenarioConfig.EntityOriented

        self.n_actions = ScenarioConfig.n_actions
        self.MaxEpisodeStep = ScenarioConfig.MaxEpisodeStep

        self.scenario_name = ScenarioConfig.scenario_name
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.scenario_path = os.path.join(current_file_path, 'JSBSim', 'configs', self.scenario_name)
        if platform.system()!="Linux":
            current_file_path = current_file_path.replace('\\', '/')
            self.scenario_name = self.scenario_name.replace('\\', '/')

        if os.path.exists(f"{self.scenario_path}.yaml"):
            if self.id == 0: print绿(f"scenario path: {self.scenario_path}.yaml")
        else:
            raise FileNotFoundError(f"scenario path: {self.scenario_path}.yaml not found")


        # other checks
        assert ScenarioConfig.EntityOriented == False,\
        'Go change the obs space in ./JSBSim/tasks/task_LAG_v2,'
        'currently is 9 + (self.num_agents - 1) * 6,'
        'entity obs vec length should be the same if we use EntityOriented'
        assert ScenarioConfig.StateProvided == True, 'yes, we provide' 

        self.reset_cnt = 0
        self._env = Env(self.scenario_name)
        self._env.reset()
        self._render_file = None

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        assert isinstance(self.action_space, gymnasium.spaces.MultiDiscrete), f"do not support action space: {self.action_space}"
        self.action_converter = MD2D(self.action_space)
        self.n_actions = self.action_converter.n

    def convert_actions(self, actions):
        if isinstance(actions, list):
            actions = np.array(actions)
        assert isinstance(actions, np.ndarray)
        if len(actions.shape) == 2:
            assert actions.shape[-1] == 1
            actions = actions.squeeze(-1)
        assert len(actions.shape) == 1
        
        assert len(actions) == sum(self.n_each_team)

        converted_actions = []
        for act_index in actions:
            converted_actions.append(
                self.action_converter.index_to_action(act_index)
            )
        
        return converted_actions

    def insert_available_act(self, info):
        next_avail_act = np.ones(shape=(self.n_each_team[self.interested_team], self.n_actions))
        info['avail-act'] = next_avail_act
        info['Avail-Act'] = next_avail_act
        return info
    
    def step(self, act):
        act = self.convert_actions(act)
    
        if self.render:
           self._set_render() 

        obs, shared_obs, rewards, dones, info = self._env.step(act)

        assert rewards.shape == (sum(self.n_each_team), 1)
        rewards = rewards.squeeze(-1)

        env_done = bool(np.all(dones, axis=None))
        current_step = info["current_step"]

        info["lag_env_done"] = env_done
        info['State'] = shared_obs
        info = self.insert_available_act(info)

        if env_done == True or current_step >= self.MaxEpisodeStep:  # 1. index N is out of bounds for axis 0 with size N
                                                                     # 2. traj.time_pointer == self._step -1 (reset() returns one more obs, and, sim time 0. returns one more obs)
                                                                     # NOTE: self._step -> 1...self.MaxEpisodeStep
            wrapped_dones = self._get_dones(True)
            
            assert len(self.n_each_team) == 2, '2 teams'
            alives = self._get_alives(dones)
            red_n = alives[0]
            blue_n = alives[1]
            red_rank = 0 if red_n > blue_n else 1
            blue_rank = 0 if blue_n > red_n else 1
            if red_rank == blue_rank:
                # info["team_ranking"] = [1, 1]
                # info["team_ranking"][self.interested_team] = 0
                info["team_ranking"] = [-1, -1]
            else:
                info["team_ranking"] = [red_rank, blue_rank]
        else: wrapped_dones = self._get_dones(False)

        return (obs, rewards, wrapped_dones, info)  # choose this if not RewardAsUnity

    def reset(self):
        info = {}
        self._render_file = None
        self.reset_cnt += 1

        if self.render:
           self._set_render() 
        
        obs, shared_obs = self._env.reset()
        # obs, shared_obs = self._dict_obs_to_list_obs(obs), self._dict_obs_to_list_obs(shared_obs)

        info["lag_env_done"] = False
        info['State'] = shared_obs
        info = self.insert_available_act(info)

        # obs: a Tensor with shape (n_agent, ...)
        # info: a dict
        return obs, info
    
    # def _dict_obs_to_list_obs(self, dict_obs):
    #     # dict obs: dict(agend_uuid, obs)
    #     list_obs = []
    #     for agent_id in self.id_each_team[self.interested_team]:
    #         list_obs.append(dict_obs[agent_id])
    #     return list_obs
    
    def _set_render(self):
        if self._render_file is not None:
            render_file = self._render_file
        else:
            render_dir = os.path.join(cfg.logdir, "JSBSim_render")
            render_file = os.path.join(
                render_dir,
                f"reset_cnt={self.reset_cnt},env_id={self.id},time={time.asctime().replace(':', '-').replace(' ', '_')}.txt.acmi"
            )
            os.makedirs(render_dir, exist_ok=True)
            self._render_file = render_file
        self._env.render(filepath=render_file)
        
    
    
    def _get_dones(self, env_done: bool):
        if env_done:
            wrapped_dones = np.ones(sum(self.n_each_team))  
        else:
            wrapped_dones = np.zeros(sum(self.n_each_team))  

        return wrapped_dones
    

    def _get_alives(self, dones) -> list:
        alives = [0, 0]
        for i in range(self.n_each_team[0]):
            if not (dones[i]):
                alives[0] += 1
        for i in range(self.n_each_team[0], sum(self.n_each_team)):
            if not (dones[i]):
                alives[1] += 1
        return alives
