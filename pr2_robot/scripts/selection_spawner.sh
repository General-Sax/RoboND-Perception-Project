#!/bin/bash
# This script safely launches ros nodes with buffer time to allow param server population

while true
do

read -p "Select scenario number from { 1, 2, 3 }: " selection

if [ "$selection" = "1" ]
then
  project=pick_place_project_scene_1.launch
  break
elif [ "$selection" = "2" ]
then
  project=pick_place_project_scene_2.launch
  break
elif [ "$selection" = "3" ]
then
  project=pick_place_project_scene_3.launch
  break
else
  echo "Warning: Not an acceptable option; choose from {1, 2, 3}."
fi

done


x-terminal-emulator -e roslaunch pr2_robot $project & sleep 15 &&
# x-terminal-emulator -e roslaunch pr2_moveit pr2_moveit.launch & sleep 20 &&
x-terminal-emulator -e rosrun pr2_robot pr2_motion & sleep 15 &&
rosrun pr2_robot oop_solution_stripdown.py $selection

# x-terminal-emulator -e rosrun pr2_robot oop_solution.py $selection
#x-terminal-emulator -e roslaunch pr2_robot pick_place_demo.launch & sleep 10 &&
#x-terminal-emulator -e rosrun pr2_robot pr2_pick_place_server
