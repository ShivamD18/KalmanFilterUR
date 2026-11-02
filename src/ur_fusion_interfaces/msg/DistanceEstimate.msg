# Output of one Kalman filter in ur_fusion/fusion_node.
#
# Every reading, from either sensor, produces one DistanceEstimate from each of
# the three filters (fused, ultrasonic_only, infrared_only) with the reading's
# timestamp. A filter that does not use that sensor only runs its prediction
# step, so the three can always be compared at the same instant.

std_msgs/Header header           # stamp: time of the reading; frame_id: sensor boresight (x = beam)

string filter_name               # "fused", "ultrasonic_only" or "infrared_only"
string trigger_sensor            # "ultrasonic" or "infrared"

float64 distance                 # [m] estimated distance to the object along the boresight
float64 distance_variance        # [m^2] P[0,0]
float64 range_rate               # [m/s] estimated rate of change of the distance
float64 range_rate_variance      # [(m/s)^2] P[1,1]

float64 measurement              # [m] the reading (NaN if this filter ignores the sensor, +-inf if out of range)
float64 measurement_variance     # [m^2] R used for the reading (NaN if not used)
float64 innovation               # [m] measurement minus predicted distance
float64 innovation_variance      # [m^2] S = H P H^T + R
float64 nis                      # normalised innovation squared, innovation^2 / S (~1 on average)
float64 kalman_gain              # distance component of K: how much the reading moved the estimate

bool measurement_used            # the reading was fused into this filter
bool measurement_rejected        # the reading failed the outlier gate
