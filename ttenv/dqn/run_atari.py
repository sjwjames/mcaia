"""
Run DQN training and testing on Atari environments using PyTorch.
"""
import argparse
import numpy as np
import datetime
import json
import os
import time
import torch

from .models import get_cnn_model
from .deepq import learn, load

from deep_adfq.logger import Logger
import envs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--prioritized', type=int, default=1)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=int(5*1e6))
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nb_warmup_steps', type=int, default = 10000)
parser.add_argument('--nb_epoch_steps', type=int, default = 50000)
parser.add_argument('--target_update_freq', type=float, default=1000)
parser.add_argument('--nb_test_steps',type=int, default = 10000)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--learning_rate_decay_factor', type=float, default=1.0)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=256)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.01)
parser.add_argument('--test_eps', type=float, default=.0)
parser.add_argument('--double_q', type=int, default=0)
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

def train(seed, save_dir):
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    save_dir_0 = os.path.join(save_dir, f'seed_{seed}')
    os.makedirs(save_dir_0, exist_ok=True)

    env = envs.make(args.env, 'atari', record=bool(args.record), directory=save_dir_0)

    nb_test_steps = args.nb_test_steps if args.nb_test_steps > 0 else None
    
    # Get observation shape - for Atari, typically (4, 84, 84) after preprocessing
    observation_shape = env.observation_space.shape
    
    # Create CNN model
    model_fn = get_cnn_model(
        observation_shape=observation_shape,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling)
    )

    act = learn(
        env,
        q_func=model_fn,
        lr=args.learning_rate,
        lr_decay_factor=args.learning_rate_decay_factor,
        lr_growth_factor=args.learning_rate_growth_factor,
        max_timesteps=args.nb_train_steps,
        buffer_size=args.buffer_size,
        exploration_fraction=args.eps_fraction,
        exploration_final_eps=args.eps_min,
        train_freq=4,
        print_freq=1000,
        checkpoint_freq=int(args.nb_train_steps/10),
        checkpoint_path=os.path.join(save_dir_0, "model.pkl"),
        learning_starts=args.nb_warmup_steps,
        target_network_update_freq=args.target_update_freq,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        double_q=bool(args.double_q),
        epoch_steps=args.nb_epoch_steps,
        eval_logger=Logger(args.env, 'atari',
                nb_test_steps=nb_test_steps, save_dir=save_dir_0,
                render=bool(args.render)),
        save_dir=save_dir_0,
        test_eps=args.test_eps,
        gpu_memory=args.gpu_memory,
        render=bool(args.render),
        device=args.device,
    )
    
    print("Saving model to model.pkl")
    act.save(os.path.join(save_dir_0, "model.pkl"))
    
    env.close()
    if args.record == 1:
        env.moviewriter.finish()

def test():
    env = envs.make(args.env, 'atari', render=bool(args.render),
                    record=bool(args.record), directory=args.log_dir)
    
    # Load the saved model
    act = load(os.path.join(args.log_dir, args.log_fname))
    
    episode_rew = 0
    t = 0
    while True:
        obs, done = env.reset(), False
        while(not done):
            if args.render:
                env.render()
                time.sleep(0.05)
            obs, rew, done, info = env.step(act(obs))
            # Reset only the environment but not the recorder
            if args.record and done:
                obs, done = env.env.reset(), False
            episode_rew += rew
            t += 1
        if info.get('ale.lives', 0) == 0:
            print(f"Episode reward {episode_rew:.2f} after {t} steps")
            episode_rew = 0
            t = 0

if __name__ == '__main__':
    if args.mode == 'train':
        save_dir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)
        json.dump(vars(args), open(os.path.join(save_dir, 'learning_prop.json'), 'w'))
        seed = args.seed
        for _ in range(args.repeat):
            print(f"===== TRAIN AN ATARI RL AGENT : SEED {seed} =====")
            train(seed, save_dir)
            seed += 1
        notes = input("Any notes for this experiment? : ")
        with open(os.path.join(save_dir, "notes.txt"), 'w') as f:
            f.write(notes)

    elif args.mode == 'test':
        test()
