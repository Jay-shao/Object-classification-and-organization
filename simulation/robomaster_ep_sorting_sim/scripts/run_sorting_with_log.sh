#!/usr/bin/env bash
set -o pipefail

workspace="${ROBOMASTER_EP_WS:-${HOME}/robomaster_ep_ws}"
log_dir="${workspace}/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/sorting_launch_${timestamp}.log"

mkdir -p "${log_dir}"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "错误：找不到 /opt/ros/humble/setup.bash" >&2
  exit 1
fi
source /opt/ros/humble/setup.bash

gazebo_setup="${HOME}/gazebo_ros_ws/install/setup.bash"
if [[ -f "${gazebo_setup}" ]]; then
  source "${gazebo_setup}"
fi

workspace_setup="${workspace}/install/setup.bash"
if [[ ! -f "${workspace_setup}" ]]; then
  echo "错误：找不到 ${workspace_setup}，请先运行 colcon build。" >&2
  exit 1
fi
source "${workspace_setup}"

echo "桌面分类仿真日志：${log_file}"
ros2 launch robomaster_ep_sorting_sim sorting.launch.py 2>&1 | tee -a "${log_file}"
exit "${PIPESTATUS[0]}"
