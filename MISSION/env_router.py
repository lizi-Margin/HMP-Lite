import_path_ref = {
    "native_gym": ("MISSION.native_gym.native_gym_config",                                     'ScenarioConfig'),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper",                                        'ScenarioConfig'),
    "hdb": ("MISSION.hdb.hdb_env_wrapper",                                                     'ScenarioConfig'), 
    "lag": ("MISSION.lag.env_wrapper",                                                         'ScenarioConfig'),
}

env_init_function_ref = {
    "native_gym": ("MISSION.native_gym.native_gym_config",                                     'env_init_function'),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper",                                        'make_sc2_env'),
    "hdb": ("MISSION.hdb.hdb_env_wrapper",                                                     'make_hdb_env'),
    "lag": ("MISSION.lag.env_wrapper",                                                         'make_env'),
}

##################################################################################################################################
##################################################################################################################################

import atexit
import importlib
import numpy as np
from uhtk.siri.utils.lprint import lprint
from uhtk.UTIL.colorful import print亮红
from uhtk.siri.utils.lprint import lprint_



# ------------------ functional utility ------------------ #
def load_ScenarioConfig(env_name):
    if env_name not in import_path_ref:
        assert False, ('need to find path of ScenarioConfig')
    import_path, ScenarioConfig = import_path_ref[env_name]
    ScenarioConfig = getattr(importlib.import_module(import_path), ScenarioConfig)
    return ScenarioConfig


# ------------------ main process ------------------ #
def make_parallel_envs(marker=''):
    from config import GlobalConfig
    from COMMUNICATION.shm_pool import SmartPool

    # init remote process first
    assert (GlobalConfig.num_threads % GlobalConfig.fold) == 0, ('Use n process to run n*m parallel threads!')
    smart_pool = SmartPool(
        fold=GlobalConfig.fold,
        proc_num=GlobalConfig.num_threads // GlobalConfig.fold,
        base_seed=GlobalConfig.seed
    )
    atexit.register(smart_pool.party_over)  # failsafe, handles shm leak
    
    # init 
    GlobalConfig.ScenarioConfig = load_ScenarioConfig(GlobalConfig.env_name)
    
    env_args_dict_list = [({
        'env_name':GlobalConfig.env_name, 
        'proc_index':i if ('test' not in marker) else -(i+1), 
        'marker':marker
    },) for i in range(GlobalConfig.num_threads)]
    envs = SuperpoolEnv(smart_pool, env_args_dict_list)

    return envs

class SuperpoolEnv(object):
    def __init__(self, process_pool, env_args_dict_list):
        self.SuperPool = process_pool
        self.num_envs = len(env_args_dict_list)
        self.env_name_marker = env_args_dict_list[0][0]['marker']
        self.env = 'env' + self.env_name_marker
        self.SuperPool.add_target(name=self.env, lam=EnvAutoReset, args_list=env_args_dict_list)
        # try:
        #     self.observation_space = self.SuperPool.exec_target(name=self.env, dowhat='get_obs_space')[0]
        #     self.action_space =      self.SuperPool.exec_target(name=self.env, dowhat='get_act_space')[0]
        # except:
        #     print亮红(lprint_(self, 'Gym Space is unable to transfer between processes, using string instead'))
        #     self.observation_space = self.SuperPool.exec_target(name=self.env, dowhat='get_obs_space_str')[0]
        #     self.action_space =      self.SuperPool.exec_target(name=self.env, dowhat='get_act_space_str')[0]

    # def get_space(self):
    #     return {'obs_space': self.observation_space, 'act_space': self.action_space}

    def step(self, actions):
        # ENV_PAUSE = [np.isnan(thread_act).any() for thread_act in actions]
        results = self.SuperPool.exec_target(name=self.env, dowhat='step', args_list=actions)
        obs, rews, dones, infos = zip(*results)

        try:
            return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)
        except:
            raise RuntimeError(lprint_(self, 'Unaligned obs/reward/done is illegal!'), obs, rews, dones)

    def reset(self):
        results = self.SuperPool.exec_target(name=self.env, dowhat='reset')
        if isinstance(results[0], tuple):
            # some envs like starcraft and unreal-hmp return (ob, info) tuple at reset, deal with it
            obs, infos = zip(*results)
            return np.stack(obs), np.stack(infos)
        else:
            # but other rather simple env like MAPE only return ob
            return np.stack(results)



# ------------------ remote process ------------------ #
def make_env_function(env_name, rank):
    from config import GlobalConfig
    assert env_name == GlobalConfig.env_name, "WTF, remote process GlobalConfig.env_name is fucked up"
    GlobalConfig.ScenarioConfig = load_ScenarioConfig(env_name)

    import_path, func_name = env_init_function_ref[env_name]
    env_init_function = getattr(importlib.import_module(import_path), func_name)
    return lambda: env_init_function(env_name, rank)


# This class execute in child process
class EnvAutoReset(object):
    def __init__(self, env_args_dict):
        env_name = env_args_dict['env_name']
        proc_index = env_args_dict['proc_index']
        env_init_fn = make_env_function(env_name=env_name, rank=proc_index)
        # finally the env is initialized
        self._env = env_init_fn()
        self._cold_start = True
        self._suffer_reset = False
        # # get the space of env
        # self.observation_space = self._env.observation_space
        # self.action_space = self._env.action_space
        self._step_cache = None
        self._reset_cache = None

    def step(self, act):
        # If we receive a skip step command, 
        # we skip by returning previous obs from cache 
        # (as an echo from the previous episode)
        if np.isnan(act).any():  
            # If any of the act is NaN, we take it as a skip command
            assert self._suffer_reset
            assert self._step_cache is not None
            return self._step_cache
        
        # other wise, we step
        ob, reward, done, info = self._env.step(act)
        # avoid returning list as observation matrix
        assert isinstance(ob, np.ndarray), "Everything should be np.ndarray, except info"
        
        if np.any(done):
            # # If the environment is terminated after step
            # # (1), put terminal obs into 'info'
            # if info is None:
            #     info = {'obs-echo':ob}
            # else:
            #     assert isinstance(info, dict), ('Info must be a python dictionary')
            #     info.update({'obs-echo': ob.copy()})

            # (2), automatically reset env
            ob = self._reset_cache = self._real_reset()
            if isinstance(ob, tuple): ob = ob[0]  # (ob, info,)
        else:
            self._suffer_reset = False
            self._reset_cache = None
            
        # preserve an echo here will be use to handle unexpected env pause
        self._step_cache = [ob, reward, done, info]
        # give everything back to main process
        return (ob, reward, done, info)

    # def dict_update(self, info, info_reset):
    #     for key in info_reset:
    #         if key in info: info[key+'-echo'] = info.pop(key)
    #     info.update(info_reset)
    #     return info

    def reset(self):
        if self._cold_start:
            # this is the first time that this env thread gets reset.
            self._cold_start = False
            return self._real_reset()
        elif self._suffer_reset:
            # we have already reset previously, avoid doing that again by returning cache
            assert self._reset_cache is not None
            return self._reset_cache
        else:
            lprint(self, 'We do not recommand resetting manually.')
            return self._real_reset()

    def _real_reset(self):
        self._suffer_reset = True
        res = self._env.reset()
        if isinstance(res, tuple):
            assert len(res) == 2
            ob, info = res
        else: ob = res
        assert isinstance(ob, np.ndarray), "Everything should be np.ndarray, except info"
        return res

    # def sleep(self):
    #     return self._env.sleep()

    def render(self):
        return self._env.render()

    def close(self):
        return None

    # def get_act_space(self):
    #     return self.action_space

    # def get_obs_space(self):
    #     return self.observation_space

    # def get_act_space_str(self):
    #     return str(self.action_space)

    # def get_obs_space_str(self):
    #     return str(self.observation_space)

    def __del__(self):
        if hasattr(self,'env'): 
            del self._env


