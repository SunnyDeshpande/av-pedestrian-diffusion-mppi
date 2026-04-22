# install
```bash
mkdir -p ~/CS588/<your_group_name>
cd ~/CS588/<your_group_name>
git clone https://github.com/hungdche/gem_ws.git
cd gem_ws
```

# compile workspace
```bash
colcon build --symlink-install 
```

# launch sensor
```bash
source install/setup.bash
ros2 launch basic_launch sensor_init.launch.py
```

# launch corner cameras
IMPORTANT: Because of the e4 incident, the corner cameras are not compatible with this backup PC, so corner cameras will not work.  
```bash
source install/setup.bash
ros2 launch basic_launch corner_cameras.launch.py
```

# launch gnss
```bash
source install/setup.bash
ros2 launch basic_launch visualization.launch.py
```

# launch joystick control
```bash
source install/setup.bash
ros2 launch basic_launch dbw_joystick.launch.py
```

# launch path tracking controller
**IMPORTANT**: 
- Make sure that the heading in GNSS control is correct. Relaunch GNSS or restart the machine if you have to 
- Make sure to disable to joystick control

On one terminal:
```bash
source install/setup.bash
ros2 launch pacmod2 pacmod2.launch.xml
```

On another monitor:
```bash
source install/setup.bash
ros2 run gem_gnss_control pure_pursuit
```

