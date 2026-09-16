import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('robomaster_ep_sorting_sim')
    description_resource_root = os.path.dirname(
        get_package_share_directory('robomaster_description')
    )
    model_file = PathJoinSubstitution(
        [package_share, 'urdf', 'robomaster_ep_sorting.urdf.xacro']
    )
    world_file = PathJoinSubstitution(
        [package_share, 'worlds', 'sorting_world.sdf']
    )
    controller_file = PathJoinSubstitution(
        [package_share, 'config', 'controllers.yaml']
    )

    robot_description = {
        'robot_description': ParameterValue(
            Command([FindExecutable(name='xacro'), ' ', model_file]),
            value_type=str,
        ),
        'use_sim_time': True,
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
            )
        ),
        launch_arguments={'gz_args': [world_file, ' -r -v 2']}.items(),
    )

    clock_and_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/sorting_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/sorting_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robomaster_ep',
            '-allow_renaming', 'false',
            '-x', '0.45', '-y', '0.0', '-z', '0.0', '-Y', '0.0',
        ],
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'ep_arm_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_file,
        ],
        output='screen',
    )

    detector = Node(
        package='robomaster_ep_sorting_sim',
        executable='grid_detector.py',
        name='grid_detector',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    task = Node(
        package='robomaster_ep_sorting_sim',
        executable='sorting_task.py',
        name='sorting_task',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            description_resource_root + os.pathsep
            + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        ),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            description_resource_root + os.pathsep
            + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
        ),
        gazebo,
        clock_and_camera_bridge,
        robot_state_publisher,
        detector,
        spawn_robot,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_robot, on_exit=[joint_state_spawner])
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[arm_controller_spawner],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=arm_controller_spawner, on_exit=[task])
        ),
    ])
