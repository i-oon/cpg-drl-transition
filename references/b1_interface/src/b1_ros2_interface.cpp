/*****************************************************************
                    B1 ROS2 Interface Node
******************************************************************/

#include "b1_interface/b1_ros2_interface.hpp"
#include <chrono>
#include <cmath>
#include <functional>

B1ROS2Interface::B1ROS2Interface(const rclcpp::NodeOptions & options)
    : rclcpp::Node("b1_interface", options)
{
    RCLCPP_INFO(this->get_logger(), "Initializing B1 ROS2 Interface...");

    // Declare all parameters
    declareParameters();

    // Initialize B1 robot interface (LOW_LEVEL control mode)
    try {
        b1_ = std::make_shared<myB1>(LOWLEVEL);
        RCLCPP_INFO(this->get_logger(), "B1 interface initialized successfully");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize B1 interface: %s", e.what());
        throw;
    }

    // Setup robot parameters
    b1_->rate = rosrate_;
    b1_->tau_max = tau_max_;
    b1_->speed_max = v_max_;
    b1_->setPD(&pidgains_);

    // Perform battery check
    b1_->BatteryCheck();

    // Create publishers (8 feedback topics)
    joint_position_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/B1/joint_position", 1);
    joint_velocity_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/B1/joint_velocity", 1);
    joint_torque_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/B1/joint_torque", 1);
    joint_mode_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("/B1/joint_mode", 1);
    joint_temp_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("/B1/joint_temp", 1);
    body_rpy_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("/B1/body_rpy", 1);
    foot_contact_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/B1/foot_contact", 1);
    battery_pub_ = this->create_publisher<sensor_msgs::msg::BatteryState>("/B1/battery", 1);

    // Create subscribers (2 command topics)
    joint_target_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/B1/joint_target", 1, std::bind(&B1ROS2Interface::jointTargetCallback, this, std::placeholders::_1));
    connection_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/B1/connection", 1, std::bind(&B1ROS2Interface::connectionCallback, this, std::placeholders::_1));

    // Service: command all joints back to the fixed home position
    go_home_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/B1/go_home",
        std::bind(&B1ROS2Interface::goHomeCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    // Create publish timer (50 Hz main loop)
    auto timer_period = std::chrono::milliseconds(20); // 50 Hz
    publish_timer_ = this->create_wall_timer(
        timer_period, std::bind(&B1ROS2Interface::publishTimerCallback, this));

    // Register parameter callback for dynamic updates
    param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&B1ROS2Interface::parametersCallback, this, std::placeholders::_1));

    // Initialize connection / watchdog timers
    last_connection_time_ = this->now();
    last_command_time_ = this->now();
    last_recv_change_time_ = this->now();

    // Start background threads
    startBackgroundThreads();

    RCLCPP_INFO(this->get_logger(), "B1 ROS2 Interface initialized - waiting for /B1/connection signal");
}

B1ROS2Interface::~B1ROS2Interface()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down B1 ROS2 Interface...");
    stopBackgroundThreads();
}

// ==================== Initialization Methods ====================

void B1ROS2Interface::declareParameters()
{
    // Control parameters
    this->declare_parameter<int>("rosrate", rosrate_);
    this->declare_parameter<bool>("running", running_.load());
    this->declare_parameter<double>("maxtorque", tau_max_);
    this->declare_parameter<double>("maxspeed", v_max_);

    // Safety watchdog parameters
    this->declare_parameter<bool>("motor_output", motor_output_);
    this->declare_parameter<double>("lan_timeout", lan_timeout_);
    this->declare_parameter<double>("command_timeout", command_timeout_);

    // Default joint position (control space) — used to fill NaN slots in joint_target.
    // Pre-filled with measured real B1 standing pose (hip=0.17, thigh=0.23, calf=-0.38).
    // Override in launch file or via ros2 param set.
    // Load home position (fixed standing pose, never changes during operation)
    const std::vector<double> standing = {
        0.17, 0.23, -0.38,   // FR hip, thigh, calf
        0.17, 0.23, -0.38,   // FL
        0.17, 0.23, -0.38,   // RR
        0.17, 0.23, -0.38    // RL
    };
    this->declare_parameter<std::vector<double>>("default_joint_positions", standing);
    auto def = this->get_parameter("default_joint_positions").as_double_array();
    if ((int)def.size() == 12) {
        for (int i = 0; i < 12; ++i) {
            home_position_[i]          = static_cast<float>(def[i]);
            joint_command_buffer_[i]   = static_cast<float>(def[i]); // working pos starts at home
        }
        RCLCPP_INFO(this->get_logger(),
            "Home position loaded — working position initialised to home.");
    } else {
        RCLCPP_WARN(this->get_logger(),
            "default_joint_positions must have 12 values — NaN fill uses zero fallback.");
    }

    // PID gains - hip
    this->declare_parameter<int>("hip.kp", pidgains_.hip.kp);
    this->declare_parameter<int>("hip.kd", pidgains_.hip.kd);
    this->declare_parameter<double>("hip.tau", pidgains_.hip.tau);

    // PID gains - thigh
    this->declare_parameter<int>("thigh.kp", pidgains_.thigh.kp);
    this->declare_parameter<int>("thigh.kd", pidgains_.thigh.kd);
    this->declare_parameter<double>("thigh.tau", pidgains_.thigh.tau);

    // PID gains - knee
    this->declare_parameter<int>("knee.kp", pidgains_.knee.kp);
    this->declare_parameter<int>("knee.kd", pidgains_.knee.kd);
    this->declare_parameter<double>("knee.tau", pidgains_.knee.tau);
}

void B1ROS2Interface::updateControlParameters()
{
    // Get current parameters
    rosrate_ = this->get_parameter("rosrate").as_int();
    running_ = this->get_parameter("running").as_bool();
    tau_max_ = this->get_parameter("maxtorque").as_double();
    v_max_ = this->get_parameter("maxspeed").as_double();

    motor_output_ = this->get_parameter("motor_output").as_bool();
    lan_timeout_ = this->get_parameter("lan_timeout").as_double();
    command_timeout_ = this->get_parameter("command_timeout").as_double();

    // Update PID gains
    pidgains_.hip.kp = this->get_parameter("hip.kp").as_int();
    pidgains_.hip.kd = this->get_parameter("hip.kd").as_int();
    pidgains_.hip.tau = this->get_parameter("hip.tau").as_double();

    pidgains_.thigh.kp = this->get_parameter("thigh.kp").as_int();
    pidgains_.thigh.kd = this->get_parameter("thigh.kd").as_int();
    pidgains_.thigh.tau = this->get_parameter("thigh.tau").as_double();

    pidgains_.knee.kp = this->get_parameter("knee.kp").as_int();
    pidgains_.knee.kd = this->get_parameter("knee.kd").as_int();
    pidgains_.knee.tau = this->get_parameter("knee.tau").as_double();

    // Apply to robot
    std::lock_guard<std::mutex> lock(b1_mutex_);
    b1_->rate = rosrate_;
    b1_->tau_max = tau_max_;
    b1_->speed_max = v_max_;

    // Detect motor_output false → true transition and reset before enabling.
    // Without this, enabling fires with t=1 (full gains) and the hip feedforward
    // (-4 Nm) jerks all 4 hips immediately even with zero position error.
    const bool prev_motor_output = b1_->motor_output;
    b1_->motor_output = motor_output_;
    if (motor_output_ && !prev_motor_output && link_alive_) {
        b1_->resetForMotorEnable();
        RCLCPP_INFO(this->get_logger(),
            "motor_output enabled — position seeded from current, soft-start reset.");
    }

    b1_->setPD(&pidgains_);
}

// ==================== ROS2 Callback Methods ====================

void B1ROS2Interface::publishTimerCallback()
{
    // Update parameters dynamically
    updateControlParameters();

    // Publish robot state
    publishRobotState();

    // Clear homing window once the robot has had time to reach home
    if (homing_) {
        const auto now = this->now();
        if ((now - homing_start_time_).seconds() >= HOMING_TIMEOUT) {
            homing_ = false;
            RCLCPP_INFO(this->get_logger(),
                "Homing complete — resuming normal command stream.");
        } else {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Homing... %.0f s remaining before sine resumes.",
                HOMING_TIMEOUT - (now - homing_start_time_).seconds());
        }
    }

    // Evaluate safety watchdogs (LAN link + command heartbeat)
    evaluateWatchdogs();
}

void B1ROS2Interface::jointTargetCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (msg->data.size() != 12) {
        RCLCPP_WARN(this->get_logger(), "Joint target message has wrong size: %zu (expected 12)",
                    msg->data.size());
        return;
    }

    // While homing, keep the heartbeat alive but ignore the incoming command —
    // the robot needs to reach home before the sine (or any commander) takes over.
    if (homing_) {
        last_command_time_ = this->now(); // prevent heartbeat watchdog from tripping
        return;
    }

    // Mark command freshness for the heartbeat watchdog
    last_command_time_ = this->now();

    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        // Merge incoming command into joint_command_buffer_:
        // - specified joints (finite values) → update working position
        // - NaN joints → keep last working position (hold, not snap to home)
        for (size_t i = 0; i < 12; ++i) {
            if (std::isfinite(msg->data[i])) {
                joint_command_buffer_[i] = msg->data[i];
            }
            // NaN → joint_command_buffer_[i] unchanged (last commanded value)
        }
        fillWithLastCommand(joint_command_buffer_); // safety: ensure no NaN reaches setMotorCommand
        new_command_available_ = true;
    }

    {
        std::lock_guard<std::mutex> lock(b1_mutex_);
        b1_->setMotorCommand(joint_command_buffer_.data());
    }
}

void B1ROS2Interface::connectionCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    connection_status_ = msg->data;
    last_connection_time_ = this->now();

    if (connection_status_) {
        // Give the command-heartbeat watchdog a fresh grace window on (re)enable
        last_command_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Connection established - B1 ready for control");
    } else {
        RCLCPP_WARN(this->get_logger(), "Connection lost");
    }
}

rcl_interfaces::msg::SetParametersResult B1ROS2Interface::parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "Parameters updated";

    for (const auto & param : parameters) {
        RCLCPP_DEBUG(this->get_logger(), "Parameter updated: %s", param.get_name().c_str());
    }

    return result;
}

// ==================== Robot State Publishing ====================

void B1ROS2Interface::publishRobotState()
{
    std::lock_guard<std::mutex> lock(b1_mutex_);

    // Joint position
    auto joint_pos_msg = std_msgs::msg::Float32MultiArray();
    joint_pos_msg.data.resize(12);
    for (int i = 0; i < 12; ++i) {
        joint_pos_msg.data[i] = b1_->getMotorPosition(i);
    }
    joint_position_pub_->publish(joint_pos_msg);

    // Joint velocity
    auto joint_vel_msg = std_msgs::msg::Float32MultiArray();
    joint_vel_msg.data.resize(12);
    for (int i = 0; i < 12; ++i) {
        joint_vel_msg.data[i] = b1_->state.motorState[i].dq;
    }
    joint_velocity_pub_->publish(joint_vel_msg);

    // Joint torque
    auto joint_torque_msg = std_msgs::msg::Float32MultiArray();
    joint_torque_msg.data.resize(12);
    for (int i = 0; i < 12; ++i) {
        joint_torque_msg.data[i] = b1_->state.motorState[i].tauEst;
    }
    joint_torque_pub_->publish(joint_torque_msg);

    // Joint mode
    auto joint_mode_msg = std_msgs::msg::Int32MultiArray();
    joint_mode_msg.data.resize(12);
    for (int i = 0; i < 12; ++i) {
        joint_mode_msg.data[i] = static_cast<int>(b1_->state.motorState[i].mode);
    }
    joint_mode_pub_->publish(joint_mode_msg);

    // Joint temperature
    auto joint_temp_msg = std_msgs::msg::Int32MultiArray();
    joint_temp_msg.data.resize(12);
    for (int i = 0; i < 12; ++i) {
        joint_temp_msg.data[i] = static_cast<int>(b1_->state.motorState[i].temperature);
    }
    joint_temp_pub_->publish(joint_temp_msg);

    // Body RPY
    auto body_rpy_msg = geometry_msgs::msg::Vector3();
    body_rpy_msg.x = b1_->state.imu.rpy[0];
    body_rpy_msg.y = b1_->state.imu.rpy[1];
    body_rpy_msg.z = b1_->state.imu.rpy[2];
    body_rpy_pub_->publish(body_rpy_msg);

    // Foot contact
    auto foot_contact_msg = std_msgs::msg::Float32MultiArray();
    foot_contact_msg.data.resize(4);
    for (int i = 0; i < 4; ++i) {
        foot_contact_msg.data[i] = static_cast<float>(b1_->state.footForce[i]);
    }
    foot_contact_pub_->publish(foot_contact_msg);

    // Battery state
    auto battery_msg = sensor_msgs::msg::BatteryState();
    battery_msg.voltage = 0.0f;
    battery_msg.current = static_cast<float>(b1_->state.bms.current) / 1000.0f;
    battery_msg.percentage = static_cast<float>(b1_->state.bms.SOC) / 100.0f;
    battery_msg.cell_voltage.resize(16);
    for (int i = 0; i < 15; ++i) {
        float cell_vol = static_cast<float>(b1_->state.bms.cell_vol[i]) / 1000.0f;
        battery_msg.cell_voltage[i] = cell_vol;
        battery_msg.voltage += cell_vol;
    }
    battery_pub_->publish(battery_msg);
}

// ==================== Default Position Service ====================

void B1ROS2Interface::goHomeCallback(
    const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
    std_srvs::srv::Trigger::Response::SharedPtr res)
{
    // Check gates — all must be open for the command to reach the robot
    std::string warnings;
    if (!running_)        warnings += " [running=false]";
    if (!motor_output_)   warnings += " [motor_output=false]";
    if (!link_alive_)     warnings += " [LAN link down]";
    if (safety_triggered_) warnings += " [safety stop latched — restart node]";

    // Set home command and arm connection regardless of gate state.
    // Gates must be open for RobotControl() to execute it.
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        joint_command_buffer_ = home_position_;
    }
    {
        std::lock_guard<std::mutex> lock(b1_mutex_);
        b1_->setMotorCommand(joint_command_buffer_.data());
    }

    // Arm connection and start homing window — incoming /B1/joint_target is
    // blocked for HOMING_TIMEOUT seconds so the robot reaches home undisturbed.
    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connection_status_ = true;
        last_connection_time_ = this->now();
        last_command_time_    = this->now();
    }
    homing_ = true;
    homing_start_time_ = this->now();

    std::string info = "Go home — homing window started (" +
                       std::to_string((int)HOMING_TIMEOUT) + "s, sine blocked until settled).";
    if (!warnings.empty()) {
        info += " WARNING — gates not fully open:" + warnings;
        res->success = false;
    } else {
        info += " All gates open — moving to home.";
        res->success = true;
    }
    res->message = info;
    RCLCPP_INFO(this->get_logger(), "%s", info.c_str());
}

void B1ROS2Interface::fillWithLastCommand(std::array<float, 12> & cmd)
{
    // Safety pass: if any slot is still NaN (shouldn't happen after merge in callback),
    // fall back to home_position_ to guarantee setMotorCommand never gets NaN.
    for (int i = 0; i < 12; ++i) {
        if (!std::isfinite(cmd[i])) {
            cmd[i] = home_position_[i];
            RCLCPP_WARN_ONCE(this->get_logger(),
                "Residual NaN in command buffer — filled with home position as safety fallback.");
        }
    }
}

// ==================== Safety Watchdogs ====================

void B1ROS2Interface::evaluateWatchdogs()
{
    const auto now = this->now();

    // ---- Gap 1: LAN / UDP-receive watchdog ----
    // The robot streams state at ~1 kHz; if udpState.RecvCount stops advancing the
    // LAN link is down. UDP gives no disconnect signal, so we detect it here.
    unsigned long long recv_count;
    {
        std::lock_guard<std::mutex> lock(b1_mutex_);
        recv_count = b1_->getRecvCount();
    }
    if (recv_count != last_recv_count_) {
        last_recv_count_ = recv_count;
        last_recv_change_time_ = now;
    }
    bool link_alive = true;
    if (lan_timeout_ > 0.0) {
        link_alive = (now - last_recv_change_time_).seconds() < lan_timeout_;
    }
    link_alive_ = link_alive;

    if (!link_alive && connection_status_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "LAN link down: no UDP from robot for > %.0f ms - actuation disabled; re-send /B1/connection to resume",
            lan_timeout_ * 1000.0);
        connection_status_ = false; // force a deliberate re-handshake before resuming
    }

    // ---- Gap 2: command / heartbeat watchdog ----
    // A live commander streams /B1/joint_target continuously. If it stops (crash,
    // killed node, dropped link), do not keep applying its last latched command.
    if (command_timeout_ > 0.0 && connection_status_ && running_) {
        const double stale = (now - last_command_time_).seconds();
        if (stale > command_timeout_) {
            RCLCPP_WARN(this->get_logger(),
                "Command stream stale: no /B1/joint_target for %.2f s (> %.2f s) - actuation disabled; re-send /B1/connection to resume",
                stale, command_timeout_);
            connection_status_ = false;
        }
    }
}

// ==================== Threading Methods ====================

void B1ROS2Interface::startBackgroundThreads()
{
    threads_running_ = true;

    udp_recv_thread_ = std::thread(&B1ROS2Interface::udpRecvThread, this);
    udp_send_thread_ = std::thread(&B1ROS2Interface::udpSendThread, this);
    control_thread_ = std::thread(&B1ROS2Interface::controlThread, this);
    safety_thread_ = std::thread(&B1ROS2Interface::safetyThread, this);

    RCLCPP_INFO(this->get_logger(), "Background threads started");
}

void B1ROS2Interface::stopBackgroundThreads()
{
    threads_running_ = false;

    if (udp_recv_thread_.joinable()) udp_recv_thread_.join();
    if (udp_send_thread_.joinable()) udp_send_thread_.join();
    if (control_thread_.joinable()) control_thread_.join();
    if (safety_thread_.joinable()) safety_thread_.join();

    RCLCPP_INFO(this->get_logger(), "Background threads stopped");
}

void B1ROS2Interface::udpRecvThread()
{
    RCLCPP_DEBUG(this->get_logger(), "UDP Recv thread started");
    while (threads_running_ && rclcpp::ok()) {
        {
            std::lock_guard<std::mutex> lock(b1_mutex_);
            b1_->UDPRecv();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1 kHz
    }
    RCLCPP_DEBUG(this->get_logger(), "UDP Recv thread stopped");
}

void B1ROS2Interface::udpSendThread()
{
    RCLCPP_DEBUG(this->get_logger(), "UDP Send thread started");
    while (threads_running_ && rclcpp::ok()) {
        {
            std::lock_guard<std::mutex> lock(b1_mutex_);
            b1_->UDPSend();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1 kHz
    }
    RCLCPP_DEBUG(this->get_logger(), "UDP Send thread stopped");
}

void B1ROS2Interface::controlThread()
{
    RCLCPP_DEBUG(this->get_logger(), "Control thread started");
    bool was_actuating = false;
    while (threads_running_ && rclcpp::ok()) {
        // Actuate only when ALL gates are open:
        //   connection established + running enabled + motor output enabled
        //   + LAN link alive + no safety stop latched
        const bool actuate = connection_status_ && running_ && b1_->motor_output
                             && link_alive_ && !safety_triggered_;
        {
            std::lock_guard<std::mutex> lock(b1_mutex_);
            if (actuate) {
                b1_->RobotControl();
            } else if (was_actuating) {
                // Deliberate operator stop (motor_output=false, running=false,
                // watchdog trip) → hold last position so robot stays put.
                // Safety violations use Damp() via cutoff() instead.
                b1_->Hold();
            }
        }
        was_actuating = actuate;
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1 kHz
    }
    RCLCPP_DEBUG(this->get_logger(), "Control thread stopped");
}

void B1ROS2Interface::safetyThread()
{
    RCLCPP_DEBUG(this->get_logger(), "Safety thread started");
    while (threads_running_ && rclcpp::ok()) {
        bool triggered = false;
        try {
            std::lock_guard<std::mutex> lock(b1_mutex_);
            b1_->SafetyCheck();
            triggered = b1_->safetyTriggered();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Safety check failed: %s", e.what());
        }
        if (triggered && !safety_triggered_) {
            safety_triggered_ = true;
            RCLCPP_ERROR(this->get_logger(),
                "SAFETY STOP latched - actuation disabled until the node is restarted");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 5 Hz
    }
    RCLCPP_DEBUG(this->get_logger(), "Safety thread stopped");
}
