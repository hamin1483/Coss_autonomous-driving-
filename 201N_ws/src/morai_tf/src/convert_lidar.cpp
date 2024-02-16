#include <ros/ros.h>
#include "sensor_msgs/LaserScan.h"

#include <algorithm>
#include <memory>

using namespace std;

class LidarConvert : public ros::NodeHandle
{
    public:
        LidarConvert()
        {
            this->lidar_sub = this->subscribe("lidar2D", 10, &LidarConvert::lidarCallback, this);
            this->lidar_pub = this->advertise<sensor_msgs::LaserScan>("scan", 10);
        }
    private:
        ros::Publisher lidar_pub;
        ros::Subscriber lidar_sub;
        bool init_flag = false;
        sensor_msgs::LaserScan scan_msg;

        void lidarCallback(const sensor_msgs::LaserScan::ConstPtr & msg)
        {
            if(!this->init_flag){
                this->scan_msg.header.frame_id = msg->header.frame_id;
                this->scan_msg.angle_min = msg->angle_min;
                this->scan_msg.angle_max = msg->angle_max;
                this->scan_msg.angle_increment = msg->angle_increment;
                this->scan_msg.time_increment = msg->time_increment;
                this->scan_msg.scan_time = msg->scan_time;
                this->scan_msg.range_min = msg->range_min;
                this->scan_msg.range_max = msg->range_max;
                this->scan_msg.ranges.resize(msg->ranges.size());
                this->init_flag = true;
            }
            this->scan_msg.header.stamp = ros::Time::now();
            std::transform(msg->ranges.begin(), msg->ranges.begin()+180,
                        scan_msg.ranges.begin()+180,[](auto value){return value;});
            std::transform(msg->ranges.begin()+180, msg->ranges.end(),
                        scan_msg.ranges.begin(),[](auto value){return value;});

            this->lidar_pub.publish(scan_msg);
        }
};

int main(int argc, char ** argv){
    ros::init(argc, argv, "lidar_convert");
    auto lidar_convert = make_shared<LidarConvert>();
    ros::spin();
}