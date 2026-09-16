cmake_minimum_required(VERSION 3.8)
project(robomaster_ep_sorting_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/grid_detector.py
  scripts/sorting_task.py
  scripts/run_sorting_with_log.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
