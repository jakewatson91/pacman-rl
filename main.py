"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment
import time

def parse():
    parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_dqn_again', action='store_true', help='whether train DQN again')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args, record_video=False):
    start_time = time.time()
    if args.train_dqn or args.train_dqn_again:
        env_name = args.env_name or 'ALE/MsPacman-v5'
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        from agent_dqn import Agent_DQN
        # from agent_a3c import Agent_A3C

        agent = Agent_DQN(env, args)
        # agent = Agent_A3C(env, args)
        agent.train()

    if args.test_dqn:
        render_mode_value = "rgb_array" if record_video else None
        env = Environment('ALE/MsPacman-v5', args, atari_wrapper=True, test=True, render_mode=render_mode_value)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=10, record_video=record_video, vid_name = args.model_name)
    print('running time:',time.time()-start_time)

if __name__ == '__main__':
    args = parse()
    run(args, record_video=args.record_video)

