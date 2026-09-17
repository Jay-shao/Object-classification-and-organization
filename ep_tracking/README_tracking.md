# EP 抓取前目标丢失 → 视觉追踪 → 跟随 → 重新抓取

把你们训练好的 YOLOv11（bottle/box）和 RoboMaster EP 真机结合起来，实现：
**目标锁定后、还没抓的瞬间被拿走 → 小车重新搜索 → 跟过去 → 重新进入可抓取范围 → 抓取。**

## 关键约束（决定了架构）

1. **摄像头是 EP 自带的**，不是 USB 摄像头，只能通过 `robomaster` 的 `read_cv2_image()` 取图。
2. **`robomaster` SDK 只支持 Python 3.8**（带按版本编译的原生库，3.9+ 多不支持），
   而 **YOLO/torch 在 Python 3.12**。两者没法合在同一个环境。

所以真机跑法拆成**两个进程**，用本机 TCP(JSON) 通信：

```
Python 3.8                        Python 3.12
robot_bridge.py  ──TCP(JSON)──▶   tracking_brain.py
  真机+摄像头+运动                    YOLO + 追踪状态机
  (robomaster)                       (torch/ultralytics)
```

## 文件结构

| 文件 | 运行环境 | 作用 |
|------|---------|------|
| `vision.py` | 3.12 | YOLO 检测封装（给一帧图，返回检测列表） |
| `tracker.py` | 3.12 | 追踪状态机（纯逻辑） |
| `robot_bridge.py` | **3.8** | 连真机、读自带摄像头、执行运动指令，TCP 服务端 |
| `tracking_brain.py` | **3.12** | 取图→YOLO→追踪→下发指令，TCP 客户端 |
| `ep_tracking_demo.py` | 单环境备用 | 若哪天 robomaster 和 YOLO 装进同一 Python，用这个单进程版 |
| `yolo_detector.py` | 仿真机 | 替换仿真 `grid_detector.py` 的 ROS2 节点 |

## 核心状态机（就是你们讨论的那个图）

```
 SEARCH ──找到目标──▶ LOCKED ──对中(error_x→0)──▶ 靠近(bbox变大)──▶ 连续确认N帧 ──▶ 抓取
   ▲                     │
   │                     └── 目标突然不见了 ──────────────────────────┘
   └──────────── TRACK（原地扫描找目标 → 找到后再对中+靠近）
```

- **对中**：`error_x = bbox中心x − 画面中心x`，正转右、负转左。
- **靠近**：用 bbox 高度占画面比例估算距离，越大越近。
- **抓前确认（CHECK_TARGET）**：`ready_frames=3`，连续 3 帧“居中且够近”才发抓取。
  你在它还没抓的瞬间把瓶子拿走，`ready_streak` 清零 → 自动回到搜索继续跟，**不会空抓**。

## 一、先在电脑上联调追踪逻辑（可选，不碰真机）

如果只是想先把追踪状态机逻辑调通，可以用 `ep_tracking_demo.py --webcam` 对着电脑摄像头
（这只验证逻辑；真机摄像头是自带的，实际跑真机见下面第二节）：

```bash
cd C:\Users\Administrator\Desktop\ep_tracking
python ep_tracking_demo.py --class bottle --webcam
```

## 二、上真机（两进程）

### 第 1 步：Python 3.8 里启动桥接进程（管真机）

在你的 Python 3.8 环境（已装 `robomaster`）里：

```bash
python robot_bridge.py                # AP 直连模式
python robot_bridge.py --sn 3JKDH2T00XXXXX   # STA 模式带序列号
```

看到 `[bridge] 真机已连接` 和 `[bridge] 监听 ... 等待 tracking_brain 连接` 就绪。

### 第 2 步：Python 3.12 里启动决策进程（跑 YOLO）

在本机 Python 3.12（已装 torch/ultralytics/cv2）里，另开一个终端：

```bash
cd C:\Users\Administrator\Desktop\ep_tracking
python tracking_brain.py --class bottle
```

如果 bridge 跑在另一台机器，加 `--host 那台机器的IP`。

### 真机需要你标定的东西

- `robot_bridge.py` 顶部 `ROT_SPEED_DEG` / `FWD_SPEED_MS`：底盘转/走多快
  （`tracking_brain.py` 顶部同名的必须一致）。
- `robot_bridge.py` 里 `grasp()` 的 `arm.moveto(x=120, y=30)`：机械臂下探/抬升坐标，按实际桌面改。
- `tracker.py` 里 `TrackerConfig` 的 `deadband_px`、`grasp_h_ratio`：画面中心死区和“够近”阈值。
- 抓取确认：把你们 `robomaster_ep_pick_place.py` 里的“闭合耗时法”（阈值 1.4s）接进
  `robot_bridge.py` 的 `grasp()`，即可判断是否真夹到，而不是盲目闭合。

## 三、仿真里换成 YOLO 检测（可选，验证接口打通）

在 Ubuntu 仿真机（`/home/xiaobai/robomaster_ep_ws`）上：

1. 把 `vision.py` 和 `yolo_detector.py` 复制进 `src/robomaster_ep_sorting_sim/scripts/`。
2. 编辑该包 `CMakeLists.txt`，在 `install(PROGRAMS ...)` 里加上 `scripts/vision.py` 和 `scripts/yolo_detector.py`。
3. 编辑 `launch/sorting.launch.py`，把 detector 的 `executable='grid_detector.py'` 改成 `executable='yolo_detector.py'`。
4. 装依赖并重新编译：
   ```bash
   sudo apt install ros-humble-cv-bridge
   cd ~/robomaster_ep_ws && colcon build --symlink-install --packages-select robomaster_ep_sorting_sim
   ```

> ⚠️ 仿真里物体是红色方块/蓝色圆柱，和真实 bottle/box 差异大，YOLO 不一定认得出。
> 这个节点主要是验证“YOLO bbox 能正确转成 Detection2DArray + 网格号”这条链路；
> 真正识别效果要在真机（真实瓶子/盒子）上才有意义。

## 与仿真代码的对应关系（怎么从 sorting_task.py 迁移到真机）

| 仿真（ROS2 + Gazebo） | 真机（robomaster SDK） |
|----------------------|------------------------|
| `set_entity_pose` / `drive_to()` 挪底盘 | `chassis.drive_speed(x, y, z, timeout=)` |
| `line_arm_to()` / `set_gripper()` 机械臂 | `arm.moveto(x, y)` / `gripper.close()` |
| `grid_detector.py` 假检测 | `vision.py` 的 YOLO 真检测 |
| `sorting_task.py` 顺序状态机 | `tracker.py` 的追踪状态机 |
| `Detection2DArray` 消息 | `List[Detection]`（Python 对象） |
| 同一进程内直接调用 | 跨进程 TCP(JSON)，拆成 bridge + brain |

## 下一步建议

1. 先用 `--webcam` 把追踪逻辑调到满意（调 `deadband_px`、`grasp_h_ratio`）。
2. 上真机时，先只开底盘对中/前进，机械臂动作注释掉，确认不撞。
3. 再接机械臂 + 你们已有的“闭合耗时法”抓取检测。
4. 分类放置（bottle→左盒 / box→右盒）可以在 `READY_TO_GRASP` 后加一段“转去对应盒子放下”，
   复用 `sorting_task.py` 里 `process_object()` 的放置逻辑思路。

## 参考

- RoboMaster SDK 官方安装文档：https://robomaster-dev.readthedocs.io/zh-cn/latest/python_sdk/installs.html
- 清华镜像安装：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple robomaster`
- SDK 源码仓库：https://github.com/dji-sdk/RoboMaster-SDK （本机直连 GitHub 不通时走 ghproxy 镜像）
