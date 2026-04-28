# pacmod2_msgs

A set of ROS messages for use with the [PACMod 2 ROS driver](https://github.com/astuff/pacmod2).
This package is a hybrid ROS1/ROS2 package which means the version history will be the same on both ROS versions.

## Installation 

Install pacmod2_msgs using our debian repository:

```sh
sudo apt install apt-transport-https
sudo sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt update
sudo apt install ros-$ROS_DISTRO-pacmod2-msgs  
```

## Versioning

Since ROS msgs are not typical software libraries, a modified version of [semver](https://semver.org) will be used to version this repo:

Given a version number MAJOR.MINOR.PATCH, increment the:

1. MAJOR version when an existing message type/name changes or is removed, OR any time an existing field in a message changes or is removed (something that could require code changes by dependent packages).
2. MINOR version when the md5sum of any existing message changes, but no dependent code changes are needed. For example, adding new fields or new message types could only require re-compilation by dependent packages, but no code changes.
3. PATCH version when there is no change to the md5sum of any existing messages. For example adding a new message type or adding comment-only changes to existing messages. No code changes or re-compilation is needed by any dependent packages.
