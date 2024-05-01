"""
Script adapted from: https://docs.ray.io/en/latest/ray-core/examples/plot_pong_example.html
Modifier: Gianluca Bencomo, Berlin Chen
Purpose: Simple RL setting for analyzing benefits of Ray Core.
Optimizes simple MLP to play Atari Pong.
"""

import os
import typer
import ray
import time

import numpy as np
import gymnasium as gym

# Forces OpenMP to use 1 single thread to prevent contention between multiple actors. 
os.environ["OMP_NUM_THREADS"] = "1"
# Tells numpy to only use one core.
os.environ["MKL_NUM_THREADS"] = "1"


def preprocess(img):
    # Image preprocessing from Karpathy
    img = img[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(float).ravel()


def process_rewards(R, gamma):
    g = np.zeros_like(R)
    running_g = 0
    for t in reversed(range(0, R.size)):
        # Reset the sum, since this was a game boundary (pong specific!)
        if R[t] != 0:
            running_g = 0
        running_g = running_g * gamma + R[t]
        g[t] = running_g
    return g


def rollout(model, env):
    observation, _ = env.reset()
    prev_frame = None
    states, hs, dlogps, drs = [], [], [], []
    done = False
    while not done:
        cur_frame = preprocess(observation)
        state = cur_frame - prev_frame if prev_frame is not None else np.zeros(80 * 80)
        prev_frame = cur_frame
        aprob, h = model.forward(state)
        # 2 == left, 3 == right
        action = 2 if np.random.uniform() < aprob else 3
        states.append(state)
        hs.append(h)
        # map action 2 -> 1, and 3 -> 0 to compare to logprobs
        y = 1 if action == 2 else 0
        # error
        dlogps.append(y - aprob)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        drs.append(reward)
    return states, hs, dlogps, drs


class Model(object):
    def __init__(self, H: int, D: int = 80 * 80):
        self.weights = {}
        self.weights["W1"] = np.random.randn(H, D) / np.sqrt(D)
        self.weights["W2"] = np.random.randn(H) / np.sqrt(H)

    def forward(self, x):
        h = np.dot(self.weights["W1"], x)
        h[h < 0] = 0  # ReLU
        logp = np.dot(self.weights["W2"], h)
        p = 1.0 / (1.0 + np.exp(-logp))
        return p, h

    def backward(self, eph, epx, epdlogp):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.weights["W2"])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        return {"W1": dW1, "W2": dW2}

    def update(self, grad_buffer, rmsprop_cache, lr, decay):
        for k, v in self.weights.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g**2
            self.weights[k] += lr * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)


def zero_grads(grad_buffer):
    """Reset the batch gradient buffer."""
    for k, v in grad_buffer.items():
        grad_buffer[k] = np.zeros_like(v)


@ray.remote(num_cpus=1)
class RolloutWorker(object):
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5")

    def compute_gradient(self, model, gamma):
        # Compute a simulation episode.
        xs, hs, dlogps, drs = rollout(model, self.env)
        reward_sum = sum(drs)
        # Vectorize the arrays.
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        # Compute the discounted reward backward through time.
        discounted_epr = process_rewards(epr, gamma)
        # Standardize the rewards to be unit normal (helps control the gradient
        # estimator variance).
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        # Modulate the gradient with advantage (the policy gradient magic
        # happens right here).
        epdlogp *= discounted_epr

        # time.sleep(5)
        # dim = 600
        # a = np.random.rand(dim, dim)
        # b = np.linalg.inv(a)

        return model.backward(eph, epx, epdlogp), reward_sum


def main(
    seed: int = 0,
    hidden: int = 200,
    gamma: float = 0.99,
    alpha: float = 1e-4,
    decay: float = 0.99,
    iterations: int = 500,
    batch_size: int = 8,
):
    
    # Make sure to run the following to initialize the ray cluster:
    # ray start --head --resources '{"node": 0}'  --num-cpus=2 --num-gpus=0
    # ray start  --resources '{"node": 1}' --num-cpus=2 --num-gpus=0 --address localhost:6379
    # see the following:
    #    https://rise.cs.berkeley.edu/blog/ray-scheduling/
    #    https://docs.ray.io/en/latest/ray-core/scheduling/resources.html

    # Note: has to be 0.0.0.0.
    # See: https://github.com/grpc/grpc/issues/9789#issuecomment-281431679
    context = ray.init(address='0.0.0.0:6379')

    print(f"dashboard url is {context.dashboard_url}")

    model = Model(H=hidden)
    actors = [RolloutWorker.options(scheduling_strategy="DEFAULT", ).remote() for _ in range(batch_size)]

    running_reward = None
    grad_buffer = {k: np.zeros_like(v) for k, v in model.weights.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.weights.items()}

    for i in range(1, 1 + iterations):
        model_id = ray.put(model)
        gradient_ids = []
        start_time = time.time()
        gradient_ids = [actor.compute_gradient.remote(model_id, gamma) for actor in actors]
        for batch in range(batch_size):
            [grad_id], gradient_ids = ray.wait(gradient_ids)
            grad, reward_sum = ray.get(grad_id)
            for k in model.weights:
                grad_buffer[k] += grad[k]
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
        end_time = time.time()
        print(
            "Batch {} computed {} rollouts in {} seconds, "
            "running mean is {}".format(
                i, batch_size, end_time - start_time, running_reward
            )
        )
        model.update(grad_buffer, rmsprop_cache, alpha, decay)
        zero_grads(grad_buffer)


if __name__ == "__main__":
    typer.run(main)
