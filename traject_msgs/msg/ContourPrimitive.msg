int32 type
std_msgs/Header header

# Pose of the primitive
geometry_msgs/Pose pose

# Orientation of the normals relative to the pose orientation (usually z up, i.e. [0,0,1])
float64[] normal

# True for counter-clockwise execution, False for clock-wise
bool ccw

# resolution in points per meter (will be rounded to the next higher integer to determine number of points
float64 res

# spirals and circles
float64 r_start
float64 r_end
float64 h_start
float64 h_end
float64 rounds  # how many times the spiral should go around

# square & rectangle
float64 a
float64 b

# contour types
int32 CIRCLE=0
int32 SQUARE=1
int32 RECTANGLE=2
int32 SPIRAL=3
