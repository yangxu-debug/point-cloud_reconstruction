## Camera Logs

### Intrinsics (`intrinsics_log.csv`)
Camera intrinsic matrix per frame, stored row-major. The `timestamp` is in seconds, relative to the start of the recording.
fx(m00) 0(m10)   px(m20)
0(m01)  fy(m11)  py(m21)
0(m02)  0(m12)   1(m22)
- `fx`, `fy`: focal length in pixels.
- `px`, `py`: principal point coordinates in pixels.
- The origin is at the center of the upper-left pixel.

### Extrinsics (`extrinsics_log.csv`)
Camera extrinsic matrix (transform) per frame. This is a 4x4 matrix representing the camera's position and orientation in world space, stored row-major. The `timestamp` is in seconds, relative to the start of the recording.