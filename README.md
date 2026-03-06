**Subscribed packages**
/camera/image_raw for Mono(-Inertial) node
/camera/left/image_raw for Stereo(-Inertial) node
/camera/right/image_raw for Stereo(-Inertial) node
/imu for Mono/Stereo/RGBD-Inertial node
/camera/rgb/image_raw and /camera/depth_registered/image_raw for RGBD nod

**Published packages**
/orb_slam3/camera_pose, left camera pose in world frame, published at camera rate
/orb_slam3/body_odom, imu-body odometry in world frame, published at camera rate
/orb_slam3/tracking_image, processed image from the left camera with key points and status text
/orb_slam3/tracked_points, all key points contained in the sliding window
/orb_slam3/all_points, all key points in the map
/orb_slam3/kf_markers, markers for all keyframes' positions
/tf, with camera and imu-body poses in world frame

