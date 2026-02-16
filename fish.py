#!/usr/bin/env python3
import argparse
import logging
import time

import cv2
import gymnasium as gym
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Button, Controller

import stardew_fisher


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stardew Valley fishing Q-learning agent.')
    parser.add_argument('--eta', type=float, default=0.628, help='Learning rate for Q-learning updates.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor for future rewards.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of fishing encounters to train for.')
    parser.add_argument('--model-load-path', type=str, default='models/batch100_fish_id.h5', help='Model path used by the environment object finder.')
    parser.add_argument('--screen-dims-path', type=str, default='models/numpy_data/screen_dims.npy', help='NumPy file containing screen dimensions.')
    parser.add_argument('--cast-wait-s', type=float, default=3.0, help='Time to wait before first cast click.')
    parser.add_argument('--hook-wait-s', type=float, default=2.5, help='Time to wait after cast click before hook sequence.')
    parser.add_argument('--hook-press-s', type=float, default=0.5, help='Duration of each hook press.')
    parser.add_argument('--hook-gap-s', type=float, default=1.0, help='Gap between hook presses.')
    parser.add_argument('--post-hook-wait-s', type=float, default=1.5, help='Wait after final hook action.')
    parser.add_argument('--scan-frames', type=int, default=20, help='Frames to scan when checking if fish minigame started.')
    parser.add_argument('--frame-delay-s', type=float, default=(1 / 30), help='Delay between frame captures.')
    parser.add_argument('--catch-diff-threshold', type=float, default=0.65, help='Pixel-similarity threshold used to detect catches.')
    parser.add_argument('--bar-diff-threshold', type=float, default=0.5, help='Pixel-similarity threshold used to detect real fish.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity.')
    parser.add_argument('--debug-exit-hook', action='store_true', help='Enable debugger on process exit (disabled by default).')
    return parser.parse_args()


def _do_hook_sequence(mouse, press_duration, gap_duration):
    mouse.press(Button.left)
    time.sleep(press_duration)
    mouse.release(Button.left)
    time.sleep(gap_duration)


def train(args):
    env = gym.make(
        'StardewFisherEnv-v0',
        model_load_path=args.model_load_path,
        screen_dims_path=args.screen_dims_path,
    )
    mouse = Controller()

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    caught_something_space = (960, 480, 980, 550)
    bar_space = (800, 305, 840, 855)

    for episode_idx in range(args.episodes):
        logger.info('Starting episode %s/%s', episode_idx + 1, args.episodes)

        mouse.release(Button.left)
        time.sleep(args.cast_wait_s)
        mouse.click(Button.left)
        time.sleep(args.hook_wait_s)

        _do_hook_sequence(mouse, args.hook_press_s, args.hook_gap_s)
        _do_hook_sequence(mouse, args.hook_press_s, args.hook_gap_s)

        mouse.press(Button.left)
        time.sleep(args.hook_press_s)
        mouse.release(Button.left)
        time.sleep(args.post_hook_wait_s)

        caught_area = np.array(ImageGrab.grab(bbox=caught_something_space))
        caught_area = cv2.cvtColor(caught_area, cv2.COLOR_BGR2RGB)
        last_caught = caught_area

        bar_area = np.array(ImageGrab.grab(bbox=bar_space))
        bar_area = cv2.cvtColor(bar_area, cv2.COLOR_BGR2RGB)
        last_bar = bar_area

        caught = False
        while not caught:
            time.sleep(args.frame_delay_s)
            caught_area = np.array(ImageGrab.grab(bbox=caught_something_space))
            caught_area = cv2.cvtColor(caught_area, cv2.COLOR_BGR2RGB)
            caught_similarity = np.sum(np.where(caught_area == last_caught, 1, 0)) / caught_area.size
            if caught_similarity < args.catch_diff_threshold:
                time.sleep(0.2)
                mouse.press(Button.left)
                time.sleep(0.1)
                mouse.release(Button.left)
                caught = True
                logger.info('Detected catch prompt with similarity %.4f', caught_similarity)
            last_caught = caught_area

        bar_area_change = []
        for _ in range(args.scan_frames):
            bar_area = np.array(ImageGrab.grab(bbox=bar_space))
            bar_area = cv2.cvtColor(bar_area, cv2.COLOR_BGR2RGB)
            similarity = np.sum(np.where(bar_area == last_bar, 1, 0)) / bar_area.size
            bar_area_change.append(similarity)
            last_bar = bar_area
            time.sleep(args.frame_delay_s)

        is_actual_fish = min(bar_area_change) <= args.bar_diff_threshold
        if not is_actual_fish:
            logger.info('No fish minigame detected (min similarity %.4f)', min(bar_area_change))
            continue

        diff, _ = env.reset()
        terminated = False
        truncated = False
        step_idx = 0

        while not (terminated or truncated):
            step_idx += 1
            exploration_noise = np.random.randn(1, env.action_space.n) * (1.0 / (step_idx + 1))
            act = np.argmax(q_table[diff] + exploration_noise)
            diff1, rew, terminated, truncated, _ = env.step(act)

            logger.info('step=%s reward=%s action=%s next_state=%s', step_idx, rew, act, diff1)
            q_table[diff][act] = q_table[diff][act] + args.eta * (
                rew + args.gamma * np.max(q_table[diff1]) - q_table[diff][act]
            )
            diff = diff1

            if terminated:
                env.second = True


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    if args.debug_exit_hook:
        import atexit
        import pdb

        atexit.register(pdb.set_trace)

    train(args)


if __name__ == '__main__':
    main()
