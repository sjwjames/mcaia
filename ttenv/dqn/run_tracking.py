"""
Run DQN training and testing on target tracking environments using PyTorch.
"""
import argparse
import datetime
import json
import os
import numpy as np
import torch
import time

import ttenv
from models import get_mlp_model, get_deepsetmlp_model
from deepq import learn, load
from logger import Logger
from ttenv.dqn import deepadfq, mcknn
from ttenv.metadata import METADATA, LEAKAGE_OBS, GAUSSIAN_OBS, METADATA_v1

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=0)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--double_q', type=int, default=0)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=5000)
parser.add_argument('--buffer_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_warmup_steps', type=int, default=100)
parser.add_argument('--nb_epoch_steps', type=int, default=100)
parser.add_argument('--target_update_freq', type=float, default=50)  # This should be smaller than epoch_steps
parser.add_argument('--checkpoint_freq', type=int, default=1000)
parser.add_argument('--nb_test_steps', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--learning_rate_decay_factor', type=float, default=1.0)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--hiddens', type=str, default='64:128:64')
parser.add_argument('--log_dir', type=str, default='./experiments')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--test_eps', type=float, default=.05)
parser.add_argument('--record', type=int, default=0)
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--gpu_memory', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--ros_log', type=int, default=0)
parser.add_argument('--map', type=str, default="emptySmall")
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--eval_type', choices=['random', 'random_zone', 'fixed'], default='random')
parser.add_argument('--init_file_path', type=str, default=".")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--num_particles', type=int, default=1000)
parser.add_argument('--im_size', type=int, default=28)
parser.add_argument('--particle_belief', type=int, default=0)
parser.add_argument('--reuse_last_init', type=int, default=0)
parser.add_argument('--act_policy', choices=['egreedy', 'bayesian'], default='egreedy')
parser.add_argument('--qval_calculation', choices=['q-network', 'mc', 'mc-greedy'], default='q-network')
parser.add_argument('--alg', choices=['adfq', 'adfq-v2'], default='adfq')
parser.add_argument('--scope', type=str, default='deepadfq')
parser.add_argument('--varth', type=float, default=1e-10)
parser.add_argument('--noise', type=float, default=1.0)
parser.add_argument('--blocked', type=int, default=0)
parser.add_argument('--n_neighbours', type=int, default=5)
parser.add_argument('--n_beliefs', type=int, default=10)
parser.add_argument('--mc_reward_mode', choices=['total', 'time'], default='time')
parser.add_argument('--post_training', type=int, default=0)

args = parser.parse_args()


def train(seed, save_dir):
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir_0 = os.path.join(save_dir, f'seed_{seed}')
    os.makedirs(save_dir_0, exist_ok=True)

    env = ttenv.make(args.env,
                     render=args.render,
                     record=args.record,
                     ros=args.ros,
                     map_name=args.map,
                     directory=args.log_dir,
                     num_targets=args.nb_targets,
                     is_training=True,
                     im_size=args.im_size,
                     T_steps=args.nb_epoch_steps,
                     n_particles=args.num_particles
                     )
    if args.qval_calculation == "q-network":
        if not args.particle_belief:
            # # Parse hidden layer sizes from string
            hiddens = [int(h) for h in args.hiddens.split(':')]

            # Create MLP model
            model_fn = get_mlp_model(
                input_dim=env.observation_space.shape[0],
                hiddens=hiddens,
                layer_norm=True
            )

            # temp code delete later
            # model_fn = get_mlp_model(
            #     input_dim=env.num_targets*3,
            #     hiddens=hiddens
            # )
        else:
            # 1 for particle weight, 2 for obstacle info, observed info per target
            # model_fn = get_deepsetmlp_model((1 + env.env.target_dim) * args.nb_targets,
            #                                 env.env.agent.dim + 2 + args.nb_targets)
            model_fn = get_deepsetmlp_model(5 * args.nb_targets, env.env.agent.dim)

        if args.act_policy == "egreedy":
            act = learn(
                env,
                q_func=model_fn,
                lr=args.learning_rate,
                lr_decay_factor=args.learning_rate_decay_factor,
                lr_growth_factor=args.learning_rate_growth_factor,
                max_timesteps=args.nb_train_steps,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                exploration_fraction=args.eps_fraction,
                exploration_final_eps=args.eps_min,
                target_network_update_freq=args.target_update_freq,
                print_freq=10,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=os.path.join(save_dir_0, "model.pkl"),
                learning_starts=args.nb_warmup_steps,
                gamma=args.gamma,
                prioritized_replay=bool(args.prioritized),
                prioritized_replay_alpha=args.prioritized_replay_alpha,
                callback=None,
                double_q=bool(args.double_q),
                epoch_steps=args.nb_epoch_steps,
                eval_logger=Logger(args.env,
                                   env_type='target_tracking',
                                   save_dir=save_dir_0,
                                   render=bool(args.render),
                                   figID=1,
                                   ros=bool(args.ros),
                                   map_name=args.map,
                                   num_targets=args.nb_targets,
                                   eval_type=args.eval_type,
                                   init_file_path=args.init_file_path),
                save_dir=save_dir_0,
                test_eps=args.test_eps,
                gpu_memory=args.gpu_memory,
                render=(bool(args.render) or bool(args.ros)),
                device=args.device,
                particle_belief=args.particle_belief,
                reuse_last_init=args.reuse_last_init,
                blocked=args.blocked
            )
        else:
            act = deepadfq.learn(
                env,
                q_func=model_fn,
                lr=args.learning_rate,
                lr_decay_factor=args.learning_rate_decay_factor,
                lr_growth_factor=args.learning_rate_growth_factor,
                max_timesteps=args.nb_train_steps,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                exploration_fraction=args.eps_fraction,
                exploration_final_eps=args.eps_min,
                target_network_update_freq=args.target_update_freq,
                checkpoint_freq=args.checkpoint_freq,
                learning_starts=args.nb_warmup_steps,
                gamma=args.gamma,
                prioritized_replay=bool(args.prioritized),
                prioritized_replay_alpha=args.prioritized_replay_alpha,
                callback=None,
                alg=args.alg,
                scope=args.scope,
                sdMin=np.sqrt(args.varth),
                noise=args.noise,
                act_policy=args.act_policy,
                epoch_steps=args.nb_epoch_steps,
                reuse_last_init=args.reuse_last_init,
                eval_logger=Logger(args.env,
                                   env_type='target_tracking',
                                   save_dir=save_dir_0,
                                   render=bool(args.render),
                                   figID=1,
                                   ros=bool(args.ros),
                                   map_name=args.map,
                                   num_targets=args.nb_targets,
                                   im_size=args.im_size,
                                   eval_type=args.eval_type,
                                   init_file_path=args.init_file_path),
                save_dir=save_dir_0,
                test_eps=args.test_eps,
                gpu_memory=args.gpu_memory,
                render=(bool(args.render) or bool(args.ros))
            )
    else:
        act = mcknn.learn(env,
                          lr=args.learning_rate,
                          lr_decay_factor=args.learning_rate_decay_factor,
                          lr_growth_factor=args.learning_rate_growth_factor,
                          max_timesteps=args.nb_train_steps,
                          buffer_size=args.buffer_size,
                          batch_size=args.batch_size,
                          exploration_fraction=args.eps_fraction,
                          exploration_final_eps=args.eps_min,
                          target_network_update_freq=args.target_update_freq,
                          print_freq=10,
                          checkpoint_freq=args.checkpoint_freq,
                          checkpoint_path=os.path.join(save_dir_0, "model.pkl"),
                          learning_starts=args.nb_warmup_steps,
                          gamma=args.gamma,
                          callback=None,
                          epoch_steps=args.nb_epoch_steps,
                          eval_logger=Logger(args.env,
                                             env_type='target_tracking',
                                             save_dir=save_dir_0,
                                             render=bool(args.render),
                                             figID=1,
                                             ros=bool(args.ros),
                                             map_name=args.map,
                                             num_targets=args.nb_targets,
                                             eval_type=args.eval_type,
                                             init_file_path=args.init_file_path),
                          save_dir=save_dir_0,
                          test_eps=args.test_eps,
                          gpu_memory=args.gpu_memory,
                          render=(bool(args.render) or bool(args.ros)),
                          device=args.device,
                          particle_belief=args.particle_belief,
                          reuse_last_init=args.reuse_last_init,
                          blocked=args.blocked,
                          n_neighbours=args.n_neighbours,
                          n_beliefs=args.n_beliefs,
                          qval_calculation=args.qval_calculation,
                          reward_mode=args.mc_reward_mode)

    print("Saving model to model.pkl")
    act.save(os.path.join(save_dir_0, "model.pkl"))

    if args.record == 1:
        env.moviewriter.finish()
    if args.post_training:
        pass

def test():
    # learning_prop = json.load(open(os.path.join(args.log_dir, 'learning_prop.json'), 'r'))
    env = ttenv.make(args.env,
                     render=args.render,
                     record=args.record,
                     ros=args.ros,
                     map_name=args.map,
                     directory=args.log_dir,
                     num_targets=args.nb_targets,
                     is_training=False,
                     im_size=args.im_size,
                     T_steps=args.nb_epoch_steps,
                     n_particles=args.num_particles
                     )

    # timelimit_env = env
    # while (not hasattr(timelimit_env, '_elapsed_steps')):
    #     timelimit_env = timelimit_env.env

    # Load the model
    act = load(os.path.join(args.log_dir, args.log_fname), {"particle_belief": args.particle_belief})

    # if args.ros_log:
    #     from ttenv.target_tracking.ros_wrapper import RosLog
    #     ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

    # Use a fixed set of initial positions if given

    seed = args.seed
    for _ in range(args.repeat):
        np.random.seed(seed)
        torch.manual_seed(seed)
        test_directory_path = args.log_dir + 'test/seed_' + str(seed) + "/"
        if not args.reuse_last_init:
            test_directory_path += "random_init/"
        test_directory_path += str(METADATA["target_speed_limit"]) + "/"+ str(METADATA["lin_dist_range_a2b"][0]) + "/"
        os.makedirs(test_directory_path, exist_ok=True)
        ep = 0
        init_pos = []
        ep_nlogdetcov = ['Episode nLogDetCov']
        time_elapsed = ['Elapsed Time (sec)']
        given_init_pose, test_init_pose = [], []
        if args.init_file_path != '.' and args.reuse_last_init:
            import pickle
            given_init_pose = pickle.load(open(args.init_file_path, "rb"))
        episode_discovery_rate_dist = []
        episode_agent_target_dist = []
        if METADATA["observation_model"] == LEAKAGE_OBS:
            particle_obs = []
        while (ep < args.nb_test_steps):  # test episode
            ep += 1
            episode_rew, nlogdetcov = 0, 0
            # if args.particle_belief:
            #     obs, terminated, truncated = env.reset(init_pose_list=given_init_pose), False, False
            #     test_init_pose.append({'agent': env.agent.state,
            #                            'targets': [env.targets[i].state for i in range(args.nb_targets)],
            #                            'belief_targets': [env.belief_targets[i].state for i in
            #                                               range(args.nb_targets)]})
            # else:
            #     obs, terminated, truncated = env.reset(init_pose_list=given_init_pose), False, False
            #     test_init_pose.append({'agent': env.agent.state,
            #                            'targets': [env.targets[i].state for i in range(args.nb_targets)],
            #                            'belief_targets': [env.belief_targets[i].state for i in
            #                                               range(args.nb_targets)]})
            obs, terminated, truncated = env.reset(init_pose_list=given_init_pose, blocked=args.blocked), False, False
            test_init_pose.append({'agent': env.agent.state,
                                   'targets': [env.targets[i].state for i in range(args.nb_targets)],
                                   'belief_targets': [env.belief_targets[i].state for i in
                                                      range(args.nb_targets)]})
            s_time = time.time()
            episode_step = 0
            while not terminated and not truncated:

                if seed == args.seed:
                    if args.render:
                        env.render(log_dir=test_directory_path)
                # if args.ros_log:
                #     ros_log.log(env)
                if args.qval_calculation == "mc" or args.qval_calculation == "mc-greedy":
                    env_info = {"action_map": env.action_map, "observation_func": env.sample_observation,
                                "agent_model": env.agent,
                                "target_belief": env.belief_targets}
                    knn_state = mcknn.project_belief(obs["agent"].cpu().numpy().squeeze(), obs["target"].cpu().numpy())
                    obs["env_info"] = env_info
                    obs["knn_state"] = knn_state
                    greedy_flag = False
                    if args.qval_calculation == "mc-greedy":
                        greedy_flag = True
                    action, q_vals = act(obs, stochastic=False, update_eps=0.0, greedy_flag=greedy_flag,
                                         episode_step=episode_step)
                else:
                    action = act(obs, stochastic=False)
                next_obs, rew, terminated, truncated, info = env.step(action)
                episode_step += 1
                obs = next_obs
                episode_rew += rew
                nlogdetcov += info['mean_nlogdetcov'] if info['mean_nlogdetcov'] else 0
            episode_discovery_rate_dist.append(np.mean([dr / args.nb_epoch_steps for dr in env.discover_cnt]))
            episode_agent_target_dist.append(np.mean(env.agent_target_dist, axis=0))
            if METADATA["observation_model"] == LEAKAGE_OBS:
                particle_obs.append(env.particles_observed)
            time_elapsed.append(time.time() - s_time)
            ep_nlogdetcov.append(nlogdetcov)
            print(f"Ep.{ep} - Episode reward: {episode_rew:.2f}, Episode nLogDetCov: {nlogdetcov:.2f}")

        np.savetxt(os.path.join(test_directory_path, 'discovery_' + str(METADATA["target_speed_limit"]) + '.csv'),
                   np.array(np.round(episode_discovery_rate_dist, 4)), delimiter=",")
        np.savetxt(os.path.join(test_directory_path, 'distance_' + str(METADATA["target_speed_limit"]) + '.csv'),
                   np.array(np.round(episode_agent_target_dist, 4)), delimiter=",")
        if METADATA["observation_model"] == LEAKAGE_OBS:
            np.savetxt(
                os.path.join(test_directory_path, 'particles_obs_' + str(METADATA["target_speed_limit"]) + '.csv'),
                np.array(particle_obs), delimiter=",")
        if args.record:
            env.moviewriter.finish()
        # if args.ros_log:
        #     ros_log.save(args.log_dir)

        import pickle
        import tabulate
        pickle.dump(test_init_pose,
                    open(os.path.join(test_directory_path, str(METADATA["target_speed_limit"]) + '_test_init_pose.pkl'),
                         'wb'))
        with open(os.path.join(test_directory_path, str(METADATA["target_speed_limit"]) + '_test_result.txt'),
                  'w') as f_result:
            f_result.write(tabulate.tabulate([ep_nlogdetcov, time_elapsed], tablefmt='presto'))
        seed += 1


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
            print(f"===== TRAIN A TARGET TRACKING RL AGENT : SEED {seed} =====")
            train(seed, save_dir)
            seed += 1
        notes = input("Any notes for this experiment? : ")
        with open(os.path.join(save_dir, "notes.txt"), 'w') as f:
            f.write(notes)
    elif args.mode == 'test':
        test()
