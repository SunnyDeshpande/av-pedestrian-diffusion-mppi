std_msgs/Header header

bool enabled                # Indicates whether any system on the PACMod is enabled or disabled.
bool override_active        # Indicates whether an override has been triggered on any system.
bool veh_can_timeout        # Indicates whether a fault is active on any system.
bool str_can_timeout        # Indicates whether the CONFIG.TXT file was read correctly
bool brk_can_timeout        # Indicates a timeout has been exceeded on the user CAN interface.
bool usr_can_timeout        # Indicates a timeout has been exceeded on the brake CAN interface.
bool user_can_read_errors   # The number of read errors encoutered by the PACMod on the user CAN interface.
