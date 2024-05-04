
# Launch Grafana (visualization) and then Prometheus (collecting logs):
# Make sure you start a ray cluster beforehand.
cd /scratch/bc2188/ray/cos518/scripts
ray metrics launch-prometheus
cd /scratch/bc2188/grafana-v10.4.2
./bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web


# Use `ps aux | grep prometheus` to find and kill prometheus process

# Then, in your local laptop run the following in separate terminals:
#  ssh bc2188@neuronic.cs.princeton.edu -L 8265:128.112.155.190:8265
#  ssh bc2188@neuronic.cs.princeton.edu -L 9090:128.112.155.190:9090
#  ssh bc2188@neuronic.cs.princeton.edu -L 3000:128.112.155.190:3000

# Open local host, then you can see the ray dashboard by going to the following on your laptop http://localhost:8265/