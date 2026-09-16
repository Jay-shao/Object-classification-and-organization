# RoboMaster EP 桌面物体分类整理仿真 第一版

本工程面向 ROS 2 Humble、Gazebo Fortress 和现有 RoboMaster EP 仿真环境。
它保留 v6.4 fixed 使用的真实 EP 底盘、并联机械臂及多连杆夹爪模型，新增
九网格、两类物体、两个分类料盒、俯视相机、检测接口、状态机和任务日志。

## 第一版完成内容

- 九个取物网格在地面上沿 Y 方向排成一条直线。
- 每个小网格尺寸为 0.18 m × 0.18 m，相邻网格中心间距 0.42 m，净间隔
  0.24 m。
- 网格 1、4、7 放置红色方块，属于 `class_1`。
- 网格 2、5、9 放置蓝色圆柱，属于 `class_2`。
- 网格 3、6、8 没有实际物体。
- 左侧红色料盒对应 `class_1`，右侧蓝色料盒对应 `class_2`；料盒占地
  0.70 m × 0.55 m，明显大于取物网格。
- 机器人逐格处理。识别到目标后先沿安全通道对齐，再向前接近、下降夹取、
  抬升并后退，然后根据类别左转或右转，前往对应料盒完成放置。
- 每个物体放置完成后，机械臂复位，小车返回中央初始点，再处理下一个网格。
- 网格线、行车通道和料盒之间留有安全间隔，料盒朝向机器人一侧不设前壁，
  避免底盘、夹爪和料盒发生明显碰撞。
- `/sorting/task_state` 发布当前任务状态。
- `~/robomaster_ep_ws/logs/sorting_task_*.csv` 保存识别、抓取、分类、空网格、
  低置信度和恢复状态。
- `~/robomaster_ep_ws/logs/sorting_launch_*.log` 保存完整 ROS 2 启动输出。
- 一个 `sorting.launch.py` 同时启动 Gazebo、俯视相机桥接、检测节点、真实 EP
  模型、控制器和任务状态机。

## 第一版视觉说明

`grid_detector.py` 已使用标准接口订阅：

```text
/sorting_camera/image     sensor_msgs/Image
```

并发布：

```text
/sorting/detections       vision_msgs/Detection2DArray
```

第一版的检测节点是确定性仿真检测器：收到相机图像后，按照仿真场景中已知的
物体类别发布检测结果。这样可以先验证“识别接口—网格判断—抓取—分类—异常
处理”完整任务链。后续替换为 OpenCV 或训练模型时，只需要替换
`grid_detector.py`，任务状态机和消息接口不需要改变。

网格 6 会收到一个置信度 0.25 的 `unknown` 弱检测，但该网格在 Gazebo 中仍然
没有物体。状态机将其作为低置信度异常跳过；网格 3 和 8 作为无检测空网格
跳过，因此第一版能够记录空网格和未识别目标两类异常路径。

## 文件位置

解压后的主要文件应位于：

```text
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/launch/sorting.launch.py
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/worlds/sorting_world.sdf
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/urdf/robomaster_ep_sorting.urdf.xacro
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/config/controllers.yaml
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/scripts/grid_detector.py
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/scripts/sorting_task.py
/home/xiaobai/robomaster_ep_ws/src/robomaster_ep_sorting_sim/scripts/run_sorting_with_log.sh
```

## 编译

压缩包放入 Ubuntu 的 `~/Downloads/` 后执行：

```bash
pkill -f "sorting.launch.py" 2>/dev/null || true
pkill -f "ign gazebo" 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true

mkdir -p ~/robomaster_ep_ws/src
cd ~/robomaster_ep_ws/src
unzip -o ~/Downloads/robomaster_ep_sorting_sim_v1.zip

cd ~/robomaster_ep_ws
rm -rf build/robomaster_ep_sorting_sim install/robomaster_ep_sorting_sim

source /opt/ros/humble/setup.bash
source ~/gazebo_ros_ws/install/setup.bash

colcon build --symlink-install --executor sequential \
  --base-paths src \
  --packages-select robomaster_description robomaster_ep_sorting_sim

source install/setup.bash
```

本压缩包不执行依赖安装。如果 `vision_msgs` 在 Ubuntu 中不存在，编译仍可能
完成，但启动检测节点会报告缺少 Python 消息模块；应先确认现有环境已经提供：

```bash
ros2 interface show vision_msgs/msg/Detection2DArray
```

## 启动完整任务

只需要一个终端：

```bash
cd ~/robomaster_ep_ws
source /opt/ros/humble/setup.bash
source ~/gazebo_ros_ws/install/setup.bash
source install/setup.bash
ros2 run robomaster_ep_sorting_sim run_sorting_with_log.sh
```

也可以直接运行 Launch：

```bash
ros2 launch robomaster_ep_sorting_sim sorting.launch.py
```

## 查看接口和日志

```bash
ros2 topic echo /sorting/task_state
ros2 topic echo /sorting/detections --once
ros2 topic hz /sorting_camera/image
ls -lht ~/robomaster_ep_ws/logs
```

正常完成后，CSV 最后一行应包含：

```text
TASK_COMPLETE ... sorted=6/6 failures=0
```

## 安全与第一版边界

- 第一版通过 Gazebo 的 `set_pose` 与 `set_pose_vector` 服务演示底盘后退、转弯
  和沿安全通道行驶；这不是麦克纳姆轮动力学控制器。
- 分类任务状态机、机械臂轨迹、检测消息和日志接口可继续复用；真机阶段需要把
  底盘位姿服务替换为 RoboMaster SDK 或 ROS 2 真机驱动命令。
- 不要同时运行旧的 `robomaster_ep_sim` 和本任务，否则两个控制器管理器可能
  使用相同名称并发生冲突。
