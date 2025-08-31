import platform

if __name__ == '__main__':
    from conf_system import init_conf_system
    init_conf_system()
    from config import GlobalConfig as cfg

    # Set numpy seed
    import numpy
    numpy.random.seed(cfg.seed)

    # Init remote process, create environments also
    from MISSION.env_router import make_parallel_envs
    envs = make_parallel_envs()
    
    # Pytorch has to be init AFTER the creation of remote process, set pytorch seed
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)

    # Prepare everything else
    from task_runner import Runner
    runner = Runner(envs=envs)
    runner.run() 


elif platform.system()!="Linux":
    # Linux uses fork for multi-processing, but Windows does not, reload config for Windows
    from conf_system import init_conf_system
    cfg = init_conf_system()
