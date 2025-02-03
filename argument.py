def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--episodes', type=int, default=100000, help='number of episodes')

    # environment hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--buffer_start', type=int, default=5000, help='buffer start size')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='learning rate for training')
    parser.add_argument('--max_buffer_size', type=int, default=10000, help='maximum buffer size')

    # network hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon for epsilon-greedy')
    parser.add_argument('--epsilon_decay_steps', type=float, default=100000, help='epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.025, help='minimum epsilon')

    # save, write, print frequency
    parser.add_argument('--update_target_net_freq', type=int, default=5000, help='update target network frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--write_freq', type=int, default=100, help='write frequency')
    parser.add_argument('--print_freq', type=int, default=1000, help='print frequency')

    #prioritized queue hyperparameters
    parser.add_argument('--prioritized_alpha', type=float, default=0.5, help='alpha for prioritized experience replay')
    parser.add_argument('--prioritized_beta', type=float, default=0.4, help='beta for prioritized experience replay')
    parser.add_argument('--prioritized_beta_increment', type=float, default=0.0001, help='beta increment for prioritized experience replay')

    #multi-step learning hyperparameters
    parser.add_argument('--n', type=int, default=3, help='n-step learning')

    #noisy hyperparameters
    parser.add_argument("--noisy_std", type=float, default=0.5)

    #distributional q-learning hyperparameters
    parser.add_argument("--num_atoms", type=int, default=51)
    parser.add_argument("--v_min", type=float, default=-10)
    parser.add_argument("--v_max", type=float, default=100)

    #risk scaling
    parser.add_argument("--risk_scaling", action='store_true', help='risk scaling or not')
    parser.add_argument("--risk_scaling_on", type=bool, default=False, help='turn risk scaling on at end of episode')    
    parser.add_argument("--risk_preference", type=float, default=1.0, help='1 indicates high risk preference')
    parser.add_argument("--n_scaling", action='store_true', help='n scaling or not')
    parser.add_argument("--max_n", type=int, default=10, help='limit n in n scaling')


    #specify locations for saving
    parser.add_argument('--data_dir', type=str, default='data/', help='directory to save data')
    parser.add_argument('--model_name', type=str, default='model.pth', help='name of model')
    parser.add_argument('--no_wandb', action='store_true', help='log to wandb or not')  

    #A3C hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_steps', type=int, default=1000000, help='maximum number of steps')
    parser.add_argument('--feature_size', type=int, default=512, help='feature size')

    #models
    parser.add_argument('--no_dueling', action='store_true', help='use dueling architecture or not')
    parser.add_argument('--no_prio_replay', action='store_true', help='use prio replay buffer or not')
    parser.add_argument('--no_nstep', action='store_true', help='use n-step architecture or not')
    parser.add_argument('--no_double', action='store_true', help='use double architecture or not')
    parser.add_argument('--no_noisy', action='store_true', help='use noisy architecture or not')
    parser.add_argument('--no_distr', action='store_true', help='use distributional architecture or not')
    parser.add_argument('--no_fill', action='store_true', help='fill buffer before training start or not')
   
    #reward shaping
    parser.add_argument('--life_penalty', type=int, default=100, help='feature size')

    return parser

# linear decay (epsiolon -epsilon_min) / epsilon step