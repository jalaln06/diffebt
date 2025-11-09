from sys import prefix
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_steps_to_success = [None] * n_inits  # Initiate memory for number of steps to success
        all_smoothness_jerk = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            # Tracker for smoothness
            episode_agent_pos = [[] for _ in range(this_n_active_envs)]

            # trackers for number of steps to success
            steps_to_success = np.full(this_n_active_envs, self.max_steps, dtype=np.int32)
            env_is_successful = np.zeros(this_n_active_envs, dtype=bool)
            step_counter = 0

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)

                # Store agent position for smoothness calculation
                # info['pos_agent'] shape is (n_envs, n_action_steps, 2)
                active_info_chunk = info[this_local_slice]

                for i in range(this_n_active_envs):
                    # env_info can be a list[dict] (normal) or a dict (if episode terminated)
                    env_info = active_info_chunk[i]
                    
                    pos_chunk_list = []
                    if isinstance(env_info, list):
                        # Normal case: iterate through list of info_dicts
                        for step_info in env_info:
                            if 'pos_agent' in step_info:
                                pos_chunk_list.append(step_info['pos_agent'])
                    elif isinstance(env_info, dict):
                        # Termination case: env_info is a single info_dict
                        if 'pos_agent' in env_info:
                            pos_chunk_list.append(env_info['pos_agent'])
                    
                    if len(pos_chunk_list) > 0:
                        # Stack all positions gathered from this chunk
                        pos_chunk_for_env_i = np.stack(pos_chunk_list)
                        episode_agent_pos[i].append(pos_chunk_for_env_i)

                # check for success
                # reward shape is (n_envs, n_action_steps)
                # we only care about the envs active in this chunk
                chunk_rewards = reward[this_local_slice] 

                # new_successes shape is (this_n_active_envs, n_action_steps)
                new_successes = (chunk_rewards >= 1.0)

                # find the first step index where success occurred *within this chunk*
                any_success_in_chunk = new_successes
                first_success_in_chunk_idx = np.zeros_like(new_successes, dtype=int)

                for i in range(this_n_active_envs):
                    # if this env hasn't been marked successful yet AND it succeeded in this chunk
                    if not env_is_successful[i] and any_success_in_chunk[i]:
                        # record the global step number (plus 1 for 1-based indexing)
                        steps_to_success[i] = step_counter + first_success_in_chunk_idx[i] + 1
                        env_is_successful[i] = True

                # Increment global step counter by the number of steps in this chunk
                step_counter += action.shape[1]

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_steps_to_success[this_global_slice] = steps_to_success[this_local_slice] # store steps to success for this chunk

            full_episode_pos = [np.concatenate(pos_list, axis=0) for pos_list in episode_agent_pos]
            episode_smoothness_jerk = [0.0] * this_n_active_envs

            for i in range(this_n_active_envs):
                pos = full_episode_pos[i] # Shape (T, 2)
                if pos.shape[0] > 3: # Need at least 4 points to calculate 3rd derivative
                    # Calculate jerk (3rd derivative of position)
                    jerk = np.diff(pos, n=3, axis=0) # Shape (T-3, 2)
                    # Calculate mean squared jerk
                    mean_squared_jerk = np.mean(np.sum(jerk**2, axis=1))
                    episode_smoothness_jerk[i] = mean_squared_jerk

            all_smoothness_jerk[this_global_slice] = episode_smoothness_jerk
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        steps_to_success_agg = collections.defaultdict(list)
        smoothness_jerk_agg = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # Steps to success metric
            steps_val = all_steps_to_success[i]
            steps_to_success_agg[prefix].append(steps_val)
            log_data[prefix+f'steps_to_success_{torch.seed}'] = int(steps_val) # Cast to int for JSON

            # Jerk metric
            smoothness_val = all_smoothness_jerk[i]
            smoothness_jerk_agg[prefix].append(smoothness_val)
            log_data[prefix+f'smoothness_jerk_{seed}'] = smoothness_val

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in steps_to_success_agg.items():
            name = prefix+'mean_steps_to_success'
            # Filter out runs that never succeeded (value == max_steps)
            successful_runs = [x for x in value if x < self.max_steps]
            if len(successful_runs) > 0:
                value = np.mean(successful_runs)
            else:
                value = np.nan # Or self.max_steps, depending on how you want to log it
            log_data[name] = value

        for prefix, value in smoothness_jerk_agg.items():
            name = prefix+'mean_smoothness_jerk'
            value = np.mean(value)
            log_data[name] = value

        return log_data
