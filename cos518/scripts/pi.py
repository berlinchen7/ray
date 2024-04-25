"""
Script adapted from: https://docs.ray.io/en/latest/ray-core/examples/monte_carlo_pi.html
Modifier: Gianluca Bencomo
Purpose: Simple baseline for comparing different schedulers and exploring Ray Core functionalities.
Calculates π using the Monte Carlo method distributed over multiple tasks.
"""

import typer
import ray
import numpy as np
import time
from tqdm.auto import tqdm

import matplotlib.pyplot as plt


@ray.remote
class ProgressActor:
    """A Ray actor that tracks and reports the progress of sampling tasks.

    Attributes:
        total_num_samples (int): Total number of samples across all tasks.
        num_samples_completed_per_task (dict): Mapping of task IDs to number of completed samples.
    """

    def __init__(self, total_num_samples: int) -> None:
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task = {}
        self.total_num_inside_per_task = {}

    def report_progress(
        self, task_id: int, num_samples_completed: int, num_inside: int
    ) -> None:
        """Updates the number of samples completed for a specific task.

        Args:
            task_id (int): The identifier of the task.
            num_samples_completed (int): The number of samples completed by the task.
        """
        self.num_samples_completed_per_task[task_id] = num_samples_completed
        self.total_num_inside_per_task[task_id] = num_inside

    def get_progress(self) -> float:
        """Calculates the total progress as the fraction of total samples completed.

        Returns:
            float: The fraction of the total number of samples that have been completed.
        """
        return (
            sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
        )

    def get_results(self) -> float:
        return (sum(self.total_num_inside_per_task.values()) * 4) / (sum(
            self.num_samples_completed_per_task.values()
        ) + 1e-10)


@ray.remote
def sampling_task(
    num_samples: int, task_id: int, progress_actor: ray.actor.ActorHandle
) -> int:
    """Performs a sampling task to estimate π using the Monte Carlo method.

    Args:
        num_samples (int): The number of samples to draw.
        task_id (int): The identifier of the task.
        progress_actor (ray.actor.ActorHandle): The progress actor handle to report progress.

    Returns:
        int: The number of samples that fall inside the unit circle.
    """
    num_inside = 0
    for i in range(num_samples):
        x = np.random.uniform(-1.0, 1.0, size=(2,))
        if np.sqrt(np.sum(x**2.0)) <= 1:
            num_inside += 1

        if (i + 1) % 100000 == 0:
            progress_actor.report_progress.remote(task_id, i + 1, num_inside)

    progress_actor.report_progress.remote(task_id, num_samples, num_inside)
    return num_inside

def plot_results(times, estimates, title="Estimate of Pi as a Function of Runtime"):
    plt.figure(figsize=(10, 5))
    plt.plot(times, estimates, label="Estimated π")
    plt.axhline(y=np.pi, color='r', linestyle='--', label="Actual π")
    plt.ylim(3.135, 3.15)
    plt.title(title)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Estimate of Pi")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(
    seed: int = 0,
    num_sampling_tasks: int = 1,
    num_samples_per_task: int = 10000000,
    scheduling_strategy: str = "DEFAULT",
    plot: bool = False
):
    """Main function to run the Monte Carlo π estimation.

    Args:
        seed (int): Seed for the random number generator.
        num_sampling_tasks (int): The number of parallel sampling tasks to run.
        num_samples_per_task (int): The number of samples per task.
    """
    context = ray.init()
    np.random.seed(seed)

    total_num_samples = num_sampling_tasks * num_samples_per_task
    progress_actor = ProgressActor.options(
        scheduling_strategy=scheduling_strategy
    ).remote(total_num_samples)
    for i in range(num_sampling_tasks):
        sampling_task.options(scheduling_strategy=scheduling_strategy).remote(num_samples_per_task, i, progress_actor)

    pi_estimates = []
    times = []
    start = time.time()
    with tqdm(total=100, desc="Calculating Pi") as pbar:
        current_progress = 0
        while True:
            progress = ray.get(progress_actor.get_progress.remote())
            update = int(progress * 100) - current_progress
            pbar.update(update)
            current_progress = int(progress * 100)
            pi = ray.get(progress_actor.get_results.remote())
            pi_estimates.append(pi)
            times.append(time.time() - start)
            if progress >= 1:
                break
            time.sleep(0.5)

    pi = ray.get(progress_actor.get_results.remote())
    print(f"Estimated value of π is: {pi}")
    print(f"Script completion time: {time.time() - start:.4f} s")

    if plot:
        plot_results(times, pi_estimates)

if __name__ == "__main__":
    typer.run(main)
