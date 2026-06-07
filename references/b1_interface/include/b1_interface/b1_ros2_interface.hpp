/*****************************************************************
                    B1 ROS2 Interface Node
******************************************************************/

#ifndef B1_ROS2_INTERFACE_HPP
#define B1_ROS2_INTERFACE_HPP

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "b1_interface/myb1.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

using namespace UNITREE_LEGGED_SDK;

class B1ROS2Interface : public rclcpp::Node
{
public:
    explicit B1ROS2Interface(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~B1ROS2Interface();

private:
    // ==================== ROS2 Components ====================

    // Publishers (feedback)
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr joint_position_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr joint_velocity_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr joint_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr joint_mode_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr joint_temp_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr body_rpy_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr foot_contact_pub_;
    rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr battery_pub_;

    // Subscribers (commands)
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr joint_target_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr connection_sub_;

    // Timer for main loop (50 Hz)
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // ==================== Callback Methods ====================

    void publishTimerCallback();
    void jointTargetCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void connectionCallback(const std_msgs::msg::Bool::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> & parameters);

    void publishRobotState();
    void evaluateWatchdogs();

    // Service: command all joints back to home_position_
    void goHomeCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr req,
        std_srvs::srv::Trigger::Response::SharedPtr res);

    // Fill NaN slots in an incoming command with joint_command_buffer_ (last command).
    // This holds the last working position — joints not mentioned keep their position.
    void fillWithLastCommand(std::array<float, 12> & cmd);

    // ==================== Threading & Control ====================

    void startBackgroundThreads();
    void stopBackgroundThreads();

    void udpRecvThread();
    void udpSendThread();
    void controlThread();
    void safetyThread();

    // ==================== Parameter Management ====================

    void declareParameters();
    void updateControlParameters();

    // ==================== Member Variables ====================

    // Robot interface
    std::shared_ptr<myB1> b1_;
    std::mutex b1_mutex_;

    // Background threads
    std::thread udp_recv_thread_;
    std::thread udp_send_thread_;
    std::thread control_thread_;
    std::thread safety_thread_;
    std::atomic<bool> threads_running_{false};

    // Connection tracking
    std::atomic<bool> connection_status_{false};
    rclcpp::Time last_connection_time_;
    std::mutex connection_mutex_;

    // Command buffer
    std::array<float, 12> joint_command_buffer_{};
    std::mutex command_mutex_;
    std::atomic<bool> new_command_available_{false};

    // Home position (control space) — fixed standing pose [0.17, 0.23, -0.38] x4.
    // Never changes during operation. Service resets working position back to this.
    std::array<float, 12> home_position_{};

    // Working position (control space) — the last fully-resolved command (no NaN).
    // Initialized from home_position_ at startup. NaN slots in /B1/joint_target are
    // filled from here so joints hold their last commanded position, not snap to home.
    // joint_command_buffer_ (below) serves this role — do not add a separate array.

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr go_home_srv_;

    // Homing window: while true, incoming /B1/joint_target is ignored so the
    // robot can reach home before the sine (or any other commander) takes over.
    std::atomic<bool> homing_{false};
    rclcpp::Time homing_start_time_;
    static constexpr double HOMING_TIMEOUT = 12.0; // s (10s soft-start + 2s settling)

    // Parameters
    int rosrate_ = 50;
    std::atomic<bool> running_{false};
    float tau_max_ = 80.0f;
    float v_max_ = 15.0f;
    PIDgains pidgains_;

    // Safety watchdog parameters
    bool motor_output_ = false;       // master gate mirrored to myB1 (false => no motor output)
    double lan_timeout_ = 0.1;        // s; no UDP from robot for longer => LAN down (<=0 disables)
    double command_timeout_ = 0.5;    // s; no /B1/joint_target for longer => commander lost (<=0 disables)

    // Safety watchdog state
    std::atomic<bool> link_alive_{true};        // false => UDP link to robot is stale
    std::atomic<bool> safety_triggered_{false}; // latched hard-stop mirror of myB1::safetyTriggered()
    unsigned long long last_recv_count_{0};     // last observed UDP RecvCount (LAN watchdog)
    rclcpp::Time last_recv_change_time_;        // when RecvCount last advanced
    rclcpp::Time last_command_time_;            // when a /B1/joint_target last arrived
};

#endif // B1_ROS2_INTERFACE_HPP
