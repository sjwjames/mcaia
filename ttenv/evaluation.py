import os

import numpy as np
import matplotlib.pyplot as plt

from ttenv.base_model import GasLeakageModel
from ttenv.metadata import METADATA


def plot_tracking_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, seed_cnt, file_dir, speed_limits,
                       distances):
    pfds_data = []
    adfq_data = []
    dqn_data = []
    mc_data = []
    mc_greedy_data = []
    for sl in speed_limits:
        for dist in distances:
            d1_res = []
            d2_res = []
            d3_res = []
            d4_res = []
            d5_res = []
            for i in range(seed_cnt):
                d1 = np.loadtxt(adfq_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d1_res = np.concatenate((d1_res, d1))
                d2 = np.loadtxt(dqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d2_res = np.concatenate((d2_res, d2))
                d3 = np.loadtxt(pfdqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d3_res = np.concatenate((d3_res, d3))
                d4 = np.loadtxt(mc_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d4_res = np.concatenate((d4_res, d4))
                d5 = np.loadtxt(
                    mc_greedy_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                    delimiter=',')
                d5_res = np.concatenate((d5_res, d5))
            adfq_data.append([np.mean(d1_res), np.std(d1_res)])
            dqn_data.append([np.mean(d2_res), np.std(d2_res)])
            pfds_data.append([np.mean(d3_res), np.std(d3_res)])
            mc_data.append([np.mean(d4_res), np.std(d4_res)])
            mc_greedy_data.append([np.mean(d5_res), np.std(d5_res)])

    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    mc_data = np.array(mc_data)
    mc_greedy_data = np.array(mc_greedy_data)
    if len(speed_limits) > 1:
        x = speed_limits
    else:
        x = distances
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(x, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
    # ax.errorbar(x, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')
    ax.errorbar(x, mc_data[:, 0], yerr=mc_data[:, 1], fmt='-o', color='c', capsize=5, label='MC')
    ax.errorbar(x, mc_greedy_data[:, 0], yerr=mc_greedy_data[:, 1], fmt='-o', color='m', capsize=5,
                label='Infotaxis')
    if len(speed_limits) > 1:
        # Add labels, title, and legend
        ax.set_xlabel('Target Speed')
    else:
        # Add labels, title, and legend
        ax.set_xlabel('Initial Distances')
    ax.set_ylabel('In Sight Rate')
    ax.set_title('In Sight Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "in-sight_tracking rate.pdf")


def plot_distancefigures(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, seed_cnt, file_dir,
                         speed_limits, distances):
    pfds_data = []
    adfq_data = []
    dqn_data = []
    mc_data = []
    mc_greedy_data = []
    speed_limit = 3.0
    distance = 40.0
    for sl in speed_limits:
        if sl == speed_limit:
            for dist in distances:
                if distance == dist:
                    d1_res = np.array([])
                    d2_res = np.array([])
                    d3_res = np.array([])
                    d4_res = np.array([])
                    d5_res = np.array([])

                    for i in range(seed_cnt):
                        d1 = np.loadtxt(adfq_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                        delimiter=',')
                        d1_res = np.vstack([d1_res, d1]) if d1_res.size else d1

                        d2 = np.loadtxt(dqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                        delimiter=',')
                        d2_res = np.vstack([d2_res, d2]) if d2_res.size else d2

                        d3 = np.loadtxt(
                            pfdqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                            delimiter=',')
                        d3_res = np.vstack([d3_res, d3]) if d3_res.size else d3

                        d4 = np.loadtxt(mc_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                        delimiter=',')
                        d4_res = np.vstack([d4_res, d4]) if d4_res.size else d4

                        d5 = np.loadtxt(
                            mc_greedy_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                            delimiter=',')
                        d5_res = np.vstack([d5_res, d5]) if d5_res.size else d5

                        # if METADATA["observation_model"]:
                        #     d1_particles = np.loadtxt(adfq_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
                        #
                        #     d2_particles = np.loadtxt(dqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
                        #
                        #     d3_particles = np.loadtxt(pfdqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')

                    adfq_data.append([np.mean(d1_res, axis=0), np.std(d1_res, axis=0)])
                    dqn_data.append([np.mean(d2_res, axis=0), np.std(d2_res, axis=0)])
                    pfds_data.append([np.mean(d3_res, axis=0), np.std(d3_res, axis=0)])
                    mc_data.append([np.mean(d4_res, axis=0), np.std(d4_res, axis=0)])
                    mc_greedy_data.append([np.mean(d5_res, axis=0), np.std(d5_res, axis=0)])
    pfds_data = np.squeeze(pfds_data)
    dqn_data = np.squeeze(dqn_data)
    adfq_data = np.squeeze(adfq_data)
    mc_data = np.squeeze(mc_data)
    mc_greedy_data = np.squeeze(mc_greedy_data)

    x = np.arange(len(dqn_data[0]))+1
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
    #             label='ADFQ')  # Use fmt for line and markers
    ax.plot(x, dqn_data[0], color='g', label='DQN')
    # ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')
    ax.errorbar(x, mc_data[0], color='c', label='MC')
    ax.errorbar(x, mc_greedy_data[0], color='m',
                label='Infotaxis')
    plt.fill_between(x, dqn_data[0] - dqn_data[1], dqn_data[0] + dqn_data[1], color='g',
                     alpha=0.2, label=None)
    plt.fill_between(x, mc_data[0] - mc_data[1], mc_data[0] + mc_data[1], color='c',
                     alpha=0.2, label=None)
    plt.fill_between(x, mc_greedy_data[0] - mc_greedy_data[1],
                     mc_greedy_data[0] + mc_greedy_data[1], color='m', alpha=0.2, label=None)

    # Add labels, title, and legend
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Distance')
    ax.set_title('Distance')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "speed_limit_" + str(speed_limit) + "_initial_dist_" + str(distance) + "_distance_plot.pdf")


def plot_discovery_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, seed_cnt, file_dir, speed_limits,
                        distances):
    pfds_data = []
    adfq_data = []
    dqn_data = []
    mc_data = []
    mc_greedy_data = []
    for sl in speed_limits:
        for dist in distances:
            d1_res = []
            d2_res = []
            d3_res = []
            d4_res = []
            d5_res = []
            for i in range(seed_cnt):
                d1 = np.loadtxt(adfq_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d1_res.append(len(d1[d1 > 0.0]) / 10.0)
                d2 = np.loadtxt(dqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d2_res.append(len(d2[d2 > 0.0]) / 10.0)
                d3 = np.loadtxt(pfdqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d3_res.append(len(d3[d3 > 0.0]) / 10.0)
                d4 = np.loadtxt(mc_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d4_res.append(len(d4[d4 > 0.0]) / 10.0)
                d5 = np.loadtxt(mc_greedy_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'discovery_' + str(sl) + '.csv',
                                delimiter=',')
                d5_res.append(len(d5[d5 > 0.0]) / 10.0)
            adfq_data.append([np.mean(d1_res), np.std(d1_res)])
            dqn_data.append([np.mean(d2_res), np.std(d2_res)])
            pfds_data.append([np.mean(d3_res), np.std(d3_res)])
            mc_data.append([np.mean(d4_res), np.std(d4_res)])
            mc_greedy_data.append([np.mean(d5_res), np.std(d5_res)])
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    mc_data = np.array(mc_data)
    mc_greedy_data = np.array(mc_greedy_data)
    if len(speed_limits) > 1:
        x = speed_limits
    else:
        x = distances
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.errorbar(x, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
    #             label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(x, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
    # ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')
    ax.errorbar(x, mc_data[:, 0], yerr=mc_data[:, 1], fmt='-o', color='c', capsize=5, label='MC')
    ax.errorbar(x, mc_greedy_data[:, 0], yerr=mc_greedy_data[:, 1], fmt='-o', color='m', capsize=5,
                label='Infotaxis')
    if len(speed_limits) > 1:
        # Add labels, title, and legend
        ax.set_xlabel('Target Speed')
    else:
        # Add labels, title, and legend
        ax.set_xlabel('Initial Distances')
    ax.set_ylabel('Discovery Rate')
    ax.set_title('Discovery Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "Discovery rate.pdf")


def plot_test_results(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, seed_cnt, file_dir, speed_limits,
                      distances):
    pfds_data_means = [[], []]
    adfq_data_means = [[], []]
    dqn_data_means = [[], []]
    mc_data_means = [[], []]
    mc_greedy_data_means = [[], []]
    pfds_data_stds = [[], []]
    adfq_data_stds = [[], []]
    dqn_data_stds = [[], []]
    mc_data_stds = [[], []]
    mc_greedy_data_stds = [[], []]
    for sl in speed_limits:
        for dist in distances:
            d1_res = [[], []]
            d2_res = [[], []]
            d3_res = [[], []]
            d4_res = [[], []]
            d5_res = [[], []]
            for i in range(seed_cnt):
                with open(adfq_dirs[i] + str(sl) + "/" + str(dist) + "/" + str(sl) + '_test_result.txt', "r") as f:
                    lines = f.readlines()
                    episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                    elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                    d1_res[0] = np.concatenate((d1_res[0], episode_values[~np.isnan(episode_values)]))
                    d1_res[1].append(elapsed_values)
                with open(dqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + str(sl) + '_test_result.txt', "r") as f:
                    lines = f.readlines()
                    episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                    elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                    d2_res[0] = np.concatenate((d2_res[0], episode_values[~np.isnan(episode_values)]))
                    d2_res[1].append(elapsed_values)
                with open(pfdqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + str(sl) + '_test_result.txt', "r") as f:
                    lines = f.readlines()
                    episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                    elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                    d3_res[0] = np.concatenate((d3_res[0], episode_values[~np.isnan(episode_values)]))
                    d3_res[1].append(elapsed_values)
                with open(mc_dirs[i] + str(sl) + "/" + str(dist) + "/" + str(sl) + '_test_result.txt', "r") as f:
                    lines = f.readlines()
                    episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                    elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                    d4_res[0] = np.concatenate((d4_res[0], episode_values[~np.isnan(episode_values)]))
                    d4_res[1].append(elapsed_values)
                with open(mc_greedy_dirs[i] + str(sl) + "/" + str(dist) + "/" + str(sl) + '_test_result.txt', "r") as f:
                    lines = f.readlines()
                    episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                    elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                    d5_res[0] = np.concatenate((d5_res[0], episode_values[~np.isnan(episode_values)]))
                    d5_res[1].append(elapsed_values)

            adfq_data_means[0].append(np.mean(d1_res[0]))
            adfq_data_means[1].append(np.mean(d1_res[1]))

            dqn_data_means[0].append(np.mean(d2_res[0]))
            dqn_data_means[1].append(np.mean(d2_res[1]))

            pfds_data_means[0].append(np.mean(d3_res[0]))
            pfds_data_means[1].append(np.mean(d3_res[1]))

            mc_data_means[0].append(np.mean(d4_res[0]))
            mc_data_means[1].append(np.mean(d4_res[1]))

            mc_greedy_data_means[0].append(np.mean(d5_res[0]))
            mc_greedy_data_means[1].append(np.mean(d5_res[1]))

            adfq_data_stds[0].append(np.std(d1_res[0]))
            adfq_data_stds[1].append(np.std(d1_res[1]))

            dqn_data_stds[0].append(np.std(d2_res[0]))
            dqn_data_stds[1].append(np.std(d2_res[1]))

            pfds_data_stds[0].append(np.std(d3_res[0]))
            pfds_data_stds[1].append(np.std(d3_res[1]))

            mc_data_stds[0].append(np.std(d4_res[0]))
            mc_data_stds[1].append(np.std(d4_res[1]))

            mc_greedy_data_stds[0].append(np.std(d5_res[0]))
            mc_greedy_data_stds[1].append(np.std(d5_res[1]))

    if len(speed_limits) > 1:
        x = np.array(speed_limits)  # positions for x-axis
        width = .2
    else:
        x = np.array(distances)  # positions for x-axis
        width = 2
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.set_xticks(x)
    axes.bar(x - 1.5 * width, adfq_data_means[0], width, yerr=adfq_data_stds[0],
             capsize=5, label="ADFQ", color="tab:red")
    axes.bar(x - .5 * width, dqn_data_means[0], width, yerr=dqn_data_stds[0],
             capsize=5, label="DQN", color="tab:green")
    # axes.bar(x, pfds_data_means[0], width, yerr=pfds_data_stds[0],
    #          capsize=5, label="DPBQN", color="tab:blue")
    axes.bar(x + .5 * width, mc_data_means[0], width, yerr=mc_data_stds[0],
             capsize=5, label="MC", color="tab:cyan")
    axes.bar(x + 1.5 * width, mc_greedy_data_means[0], width, yerr=mc_greedy_data_stds[0],
             capsize=5, label="Infotaxis", color="tab:purple")

    if len(speed_limits) > 1:
        axes.set_xlabel("Target Speed Limit")
    else:
        axes.set_xlabel("Initial Distances")
    axes.set_ylabel("LogDetCov")
    axes.set_title("Episode LogDetCov Across Methods")
    axes.legend()
    plt.savefig(file_dir + "LogDetCov.pdf")

    fig, axes = plt.subplots(figsize=(8, 6), )
    axes.bar(x - 1.5 * width, adfq_data_means[1], width, yerr=adfq_data_stds[1],
             capsize=5, label="ADFQ", color="tab:red")
    axes.bar(x - .5 * width, dqn_data_means[1], width, yerr=dqn_data_stds[1],
             capsize=5, label="DQN", color="tab:green")
    # axes.bar(x, pfds_data_means[1], width, yerr=pfds_data_stds[1],
    #          capsize=5, label="DPBQN", color="tab:blue")

    axes.bar(x + 0.5 * width, mc_data_means[1], width, yerr=mc_data_stds[1],
             capsize=5, label="MC", color="tab:cyan")

    axes.bar(x + 1.5 * width, mc_greedy_data_means[1], width, yerr=mc_greedy_data_stds[1],
             capsize=5, label="Infotaxis", color="tab:purple")
    if len(speed_limits) > 1:
        axes.set_xlabel("Target Speed Limit")
    else:
        axes.set_xlabel("Initial Distances")
    axes.set_ylabel("Runtime")
    axes.set_title("Runtime across categories")
    axes.legend()

    plt.savefig(file_dir + "runtime.pdf")


def plot_distance_tracking_figures(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, seed_cnt, file_dir,
                                   speed_limits, distances):
    pfds_data = []
    adfq_data = []
    dqn_data = []
    mc_data = []
    mc_greedy_data = []
    d = 3.0
    for sl in speed_limits:
        for dist in distances:
            d1_res = []
            d2_res = []
            d3_res = []
            d4_res = []
            d5_res = []
            d1_rate_res = []
            d2_rate_res = []
            d3_rate_res = []
            d4_rate_res = []
            d5_rate_res = []

            for i in range(seed_cnt):
                d1 = np.loadtxt(adfq_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                delimiter=',')
                d1_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d1]) / 10.0)
                d1_rate_res += [len(item[item < d]) / 100.0 for item in d1]

                d2 = np.loadtxt(dqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                delimiter=',')
                d2_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d2]) / 10.0)
                d2_rate_res += [len(item[item < d]) / 100.0 for item in d2]

                d3 = np.loadtxt(pfdqn_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                delimiter=',')
                d3_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d3]) / 10.0)
                d3_rate_res += [len(item[item < d]) / 100.0 for item in d3]

                d4 = np.loadtxt(mc_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                delimiter=',')
                d4_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d4]) / 10.0)
                d4_rate_res += [len(item[item < d]) / 100.0 for item in d4]

                d5 = np.loadtxt(mc_greedy_dirs[i] + str(sl) + "/" + str(dist) + "/" + 'distance_' + str(sl) + '.csv',
                                delimiter=',')
                d5_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d5]) / 10.0)
                d5_rate_res += [len(item[item < d]) / 100.0 for item in d5]

                # if METADATA["observation_model"]:
                #     d1_particles = np.loadtxt(adfq_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
                #
                #     d2_particles = np.loadtxt(dqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
                #
                #     d3_particles = np.loadtxt(pfdqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')

            adfq_data.append([np.mean(d1_res), np.std(d1_res), np.mean(d1_rate_res), np.std(d1_rate_res)])
            dqn_data.append([np.mean(d2_res), np.std(d2_res), np.mean(d2_rate_res), np.std(d2_rate_res)])
            pfds_data.append([np.mean(d3_res), np.std(d3_res), np.mean(d3_rate_res), np.std(d3_rate_res)])
            mc_data.append([np.mean(d4_res), np.std(d4_res), np.mean(d4_rate_res), np.std(d4_rate_res)])
            mc_greedy_data.append([np.mean(d5_res), np.std(d5_res), np.mean(d5_rate_res), np.std(d5_rate_res)])
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    mc_data = np.array(mc_data)
    mc_greedy_data = np.array(mc_greedy_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(speed_limits) > 1:
        ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                    label='ADFQ')  # Use fmt for line and markers
        ax.errorbar(speed_limits, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
        # ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')
        ax.errorbar(speed_limits, mc_data[:, 0], yerr=mc_data[:, 1], fmt='-o', color='c', capsize=5, label='MC')
        ax.errorbar(speed_limits, mc_greedy_data[:, 0], yerr=mc_greedy_data[:, 1], fmt='-o', color='m', capsize=5,
                    label='Infotaxis')

        # Add labels, title, and legend
        ax.set_xlabel('Target Speed')
        ax.set_ylabel('Discovery Rate')
        ax.set_title('Discovery Rate')
        ax.legend()  # Show the legend

        # Show the plot
        plt.savefig(file_dir + "distance_discovery_rate_plot.pdf")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(speed_limits, adfq_data[:, 2], yerr=adfq_data[:, 3], fmt='-o', color='r', capsize=5,
                    label='ADFQ')  # Use fmt for line and markers
        ax.errorbar(speed_limits, dqn_data[:, 2], yerr=dqn_data[:, 3], fmt='-o', color='g', capsize=5, label='DQN')
        # ax.errorbar(speed_limits, pfds_data[:, 2], yerr=pfds_data[:, 3], fmt='-o', color='b', capsize=5, label='DPBQN')
        ax.errorbar(speed_limits, mc_data[:, 2], yerr=mc_data[:, 3], fmt='-o', color='c', capsize=5, label='MC')
        ax.errorbar(speed_limits, mc_greedy_data[:, 2], yerr=mc_greedy_data[:, 3], fmt='-o', color='m', capsize=5,
                    label='Infotaxis')

        # Add labels, title, and legend
        ax.set_xlabel('Target Speed')
        ax.set_ylabel('Tracking Rate')
        ax.set_title('Tracking Rate')
        ax.legend()  # Show the legend

        # Show the plot
        plt.savefig(file_dir + "distance_tracking_rate_plot.pdf")
    else:
        ax.errorbar(distances, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                    label='ADFQ')  # Use fmt for line and markers
        ax.errorbar(distances, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
        # ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')
        ax.errorbar(distances, mc_data[:, 0], yerr=mc_data[:, 1], fmt='-o', color='c', capsize=5, label='MC')
        ax.errorbar(distances, mc_greedy_data[:, 0], yerr=mc_greedy_data[:, 1], fmt='-o', color='m', capsize=5,
                    label='Infotaxis')

        # Add labels, title, and legend
        ax.set_xlabel('Initial Distances')
        ax.set_ylabel('Discovery Rate')
        ax.set_title('Discovery Rate')
        ax.legend()  # Show the legend

        # Show the plot
        plt.savefig(file_dir + "distance_discovery_rate_plot.pdf")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(distances, adfq_data[:, 2], yerr=adfq_data[:, 3], fmt='-o', color='r', capsize=5,
                    label='ADFQ')  # Use fmt for line and markers
        ax.errorbar(distances, dqn_data[:, 2], yerr=dqn_data[:, 3], fmt='-o', color='g', capsize=5, label='DQN')
        # ax.errorbar(speed_limits, pfds_data[:, 2], yerr=pfds_data[:, 3], fmt='-o', color='b', capsize=5, label='DPBQN')
        ax.errorbar(distances, mc_data[:, 2], yerr=mc_data[:, 3], fmt='-o', color='c', capsize=5, label='MC')
        ax.errorbar(distances, mc_greedy_data[:, 2], yerr=mc_greedy_data[:, 3], fmt='-o', color='m', capsize=5,
                    label='Infotaxis')

        # Add labels, title, and legend
        ax.set_xlabel('Initial Distances')
        ax.set_ylabel('Tracking Rate')
        ax.set_title('Tracking Rate')
        ax.legend()  # Show the legend

        # Show the plot
        plt.savefig(file_dir + "distance_tracking_rate_plot.pdf")


def plot_nonmarkovian(markov_dirs, non_markov_dirs, seed_cnt, file_dir):
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    markov_data = []
    non_markov_data = []

    for sl in speed_limits:
        d1_res = []
        d2_res = []
        d1_discovery_res = []
        d2_discovery_res = []
        d1_value = []
        d2_value = []
        for i in range(seed_cnt):
            d1 = np.loadtxt(markov_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d1_res = np.concatenate((d1_res, d1))
            d1_discovery_res.append(len(d1[d1 > 0.0]) / 10.0)
            with open(markov_dirs[i] + str(sl) + '_test_result.txt', "r") as f:
                lines = f.readlines()
                episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                d1_value = np.concatenate((d1_value, episode_values[~np.isnan(episode_values)]))
            d2 = np.loadtxt(non_markov_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d2_res = np.concatenate((d2_res, d2))
            d2_discovery_res.append(len(d2[d2 > 0.0]) / 10.0)
            with open(non_markov_dirs[i] + str(sl) + '_test_result.txt', "r") as f:
                lines = f.readlines()
                episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                d2_value = np.concatenate((d2_value, episode_values[~np.isnan(episode_values)]))

        markov_data.append(
            [np.mean(d1_res), np.std(d1_res), np.mean(d1_discovery_res), np.std(d1_discovery_res), np.mean(d1_value),
             np.std(d1_value)])
        non_markov_data.append(
            [np.mean(d2_res), np.std(d2_res), np.mean(d2_discovery_res), np.std(d2_discovery_res), np.mean(d2_value),
             np.std(d2_value)])

    markov_data = np.array(markov_data)
    non_markov_data = np.array(non_markov_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, markov_data[:, 0], yerr=markov_data[:, 1], fmt='-o', color='r', capsize=5,
                label='Markov states')  # Use fmt for line and markers
    ax.errorbar(speed_limits, non_markov_data[:, 0], yerr=non_markov_data[:, 1], fmt='-o', color='g', capsize=5,
                label='non-Markov states')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('In Sight Rate')
    ax.set_title('In Sight Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "in-sight_tracking rate.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, markov_data[:, 2], yerr=markov_data[:, 3], fmt='-o', color='r', capsize=5,
                label='Markov states')  # Use fmt for line and markers
    ax.errorbar(speed_limits, non_markov_data[:, 2], yerr=non_markov_data[:, 3], fmt='-o', color='g', capsize=5,
                label='non-Markov states')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Discovery Rate')
    ax.set_title('Discovery Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "Discovery rate.pdf")

    width = 0.2
    x = np.arange(len(speed_limits))  # positions for x-axis
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.bar(x - width, markov_data[:, 4], width, yerr=markov_data[:, 5],
             capsize=5, label="Markov states", color="tab:red")
    axes.bar(x, non_markov_data[:, 4], width, yerr=non_markov_data[:, 5],
             capsize=5, label="non-Markov states", color="tab:green")

    axes.set_xlabel("Target speed limit")
    axes.set_ylabel("Negative LogDetCov")
    axes.set_title("Episode Negative LogDetCov across methods")
    axes.legend()
    plt.savefig(file_dir + "LogDetCov.pdf")


if __name__ == '__main__':
    task = "heteroscedastic_obstacle"
    random_init = False
    # speed_limits = [0.1]
    # distances = [10.0, 20.0, 30.0, 40.0]
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    distances = [40.0]
    file_dir = "dqn/experiments/final_results/" + task + "/"
    seeds = [0, 1, 2, 3, 4]
    if task != "non-markovian":
        adfq_dirs = [os.path.join(file_dir, "TargetTracking-v1_10061353/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        dqn_dirs = [os.path.join(file_dir, "TargetTracking-v1_09242144/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        pfdqn_dirs = [os.path.join(file_dir, "TargetTracking-v1_09242144/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        mc_dirs = [os.path.join(file_dir, "TargetTracking-v1_1_10022340/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        mc_greedy_dirs = [os.path.join(file_dir, "TargetTracking-v1_1_09111117/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        plot_tracking_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, len(seeds), file_dir, speed_limits,
                           distances)
        # plot_distance(adfq_dirs,dqn_dirs,pfdqn_dirs,len(seeds))
        plot_discovery_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, len(seeds), file_dir,
                            speed_limits, distances)
        plot_test_results(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, len(seeds), file_dir, speed_limits,
                          distances)
        plot_distance_tracking_figures(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, len(seeds), file_dir,
                                       speed_limits, distances)
        # plot_distancefigures(adfq_dirs, dqn_dirs, pfdqn_dirs, mc_dirs, mc_greedy_dirs, len(seeds), file_dir,
        #                      speed_limits, distances)
    else:
        markov_dirs = [os.path.join(file_dir, "TargetTracking-v1_1_07180912/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        non_markov_dirs = [os.path.join(file_dir, "TargetTracking-v1_1_07152337/seed_0/test/seed_" + str(seed) + "/" + (
            "random_init/" if random_init else "")) for seed in seeds]
        plot_nonmarkovian(markov_dirs, non_markov_dirs, len(seeds), file_dir)
