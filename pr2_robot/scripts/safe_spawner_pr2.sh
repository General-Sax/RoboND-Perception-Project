#! /bin/bash
# This script safely launches ros nodes with buffer time to allow param server population

rm 1.breadcrumb 2.breadcrumb 3.breadcrumb

while true
do

read -p "Select scenario number from {1, 2, 3}: " selection

if [ "$selection" = "1" ]
then
  project=pick_place_project_test1.launch
  break
elif [ "$selection" = "2" ]
then
  project=pick_place_project_test2.launch
  break
elif [ "$selection" = "3" ]
then
  project=pick_place_project_test3.launch
  break
else
  echo "Warning: Not an acceptable option; choose from {1, 2, 3}."
fi

done

touch "$selection".breadcrumb

x-terminal-emulator -e roslaunch pr2_robot $project & sleep 10 &&
x-terminal-emulator -e roslaunch pr2_moveit pr2_moveit.launch & sleep 20 &&
x-terminal-emulator -e rosrun pr2_robot pr2_motion
