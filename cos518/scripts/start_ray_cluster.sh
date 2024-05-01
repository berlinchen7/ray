# Script to deploy ray cluster on a remote cluster

NUM_CPU_HEAD=0
NUM_GPU_HEAD=0

NUM_CPU_WORKER=(32)
NUM_GPU_WORKER=(0)

printf 'Starting head node with %s cpus and %s gpus\n' "${NUM_CPU_HEAD}" "${NUM_GPU_HEAD}"
ray start --head --resources '{"headnode": 1}'  --num-cpus=${NUM_CPU_HEAD} --num-gpus=${NUM_GPU_HEAD}

for i in "${!NUM_CPU_WORKER[@]}"; do
   printf 'Starting worker node %d with %s cpus and %s gpus\n' "$i" "${NUM_CPU_WORKER[i]}" "${NUM_GPU_WORKER[i]}"
   RESOURCE_NAME=$(printf '{"workernode": %d}' "1")
   ray start  --resources "$RESOURCE_NAME" --num-cpus=${NUM_CPU_WORKER[i]} --num-gpus=${NUM_GPU_WORKER[i]} --address 0.0.0.0:6379
done