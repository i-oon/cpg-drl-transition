/*****************************************************************
                B1 Interface Main Entry Point (ROS2)
******************************************************************/

#include "rclcpp/rclcpp.hpp"
#include "b1_interface/b1_ros2_interface.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<B1ROS2Interface>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}
