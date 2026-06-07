/*****************************************************************
                                myb1
******************************************************************/

#ifndef MYB1

#define MYB1


/* -----------------------------------------------------------------
                            import libraries
------------------------------------------------------------------ */
// standard libraries
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <array>

// unitree datatype
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/joystick.h"
#include "unitree_legged_sdk/b1_const.h"

// define namespace
using namespace std;
using namespace UNITREE_LEGGED_SDK;

/* -----------------------------------------------------------------
                            define variables
------------------------------------------------------------------ */

// state id
# define DT 0.001
# define TMAX 10
# define TSTEP 1
# define PI 3.1415926

# define CUTOFF_TEMP 80 // 80 cs
# define CUTOFF_UPPERLIM 0.8 // 0.8 rads
# define CUTOFF_LOWERLIM -1.2 // -1.2 rads
//# define CUTOFF_SPEED 15.7 // rads/sec

/* -----------------------------------------------------------------
                        unitree B1 interface class
------------------------------------------------------------------ */

class myB1
{
public:
    myB1(uint8_t level) : safe(LeggedType::B1),
                            udp(level, 8090, "192.168.123.10", 8007)
    {
        udp.InitCmdData(cmd);
        UDPRecv();
    }

    // standard unitree protocol
    void UDPRecv();
    void UDPSend();

    //customize interface
    void setMotorCommand(float newmotorcommand[]);
    void setPD(PIDgains* pidgains);

    // control loop
    void RobotControl();

    // get & set methods
    float getMotorPosition(int id);

    // safety function
    void cutoff(int status);
    void SafetyCheck();
    void BatteryCheck();
    void Damp();                       // passive damping — safety violations only (robot goes limp)
    void Hold();                       // hold last position — deliberate operator stops
    bool safetyTriggered();            // true once a hard safety stop has latched

    // Call every time motor_output transitions false → true.
    // Re-seeds commanded position from current measured state and resets the
    // soft-start ramp so gains rise gradually instead of snapping to full force.
    // Prevents the hip-feedforward (-4 Nm) jerk on re-enable.
    void resetForMotorEnable();

    // link / output status helpers
    unsigned long long getRecvCount(); // total UDP packets received from robot (LAN watchdog)

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};

    // joystick command
    xRockerBtnDataStruct jointStickCommand;

    //float t = 0; // time
    float rate = 10; // update rate (rosrate)
    float tau_max = 15; // max torque/control input (clip)
    float speed_max = 15.7; // max joint speed (cutoff)
    float dt = DT; // control loop dt
    float ti = 0; // dti = rosrate/dt

    bool motor_output = false; // master gate: if false, NOTHING is ever sent to the motors
    float damp_kd = 3.0; // velocity damping gain used by Damp() on a safety/comms cutoff

private:

    bool safety_triggered = false; // latches true on a hard safety violation (cleared only by restart)
    bool control_initialized = false; // latches true after first RobotControl() run

    float t = 0; // timestep for soft start
    float intp_t = 0; // interpolation time/variable
    float tau[12] = {0}; // grav com
    float kp[12] = {100}; // P-gain
    float kd[12] = {10}; // D-gain

    const char* legname[4] = {"RF","LF","RH","LH"}; // motor name

    const float offset[12] = {      -0.2,0.85,-1.56,
                                    0.2,0.85,-1.56,
                                    -0.2,0.85,-1.56,
                                    0.2,0.85,-1.56   }; // motor offset

    /*
    const float offset[12] = {      -0.2,0.65,-1.3,
                                    0.2,0.65,-1.3,
                                    -0.2,0.65,-1.3,
                                    0.2,0.65,-1.3   }; // motor offset
    */

    const float direction[12] = {   1.0,1.0,1.0,
                                    -1.0,1.0,1.0,
                                    1.0,1.0,1.0,
                                    -1.0,1.0,1.0    }; // motor direction

    const float ucommandlimit[12] = {   b1_Hip_max,b1_Thigh_max,b1_Calf_max,
                                        b1_Hip_max,b1_Thigh_max,b1_Calf_max,
                                        b1_Hip_max,b1_Thigh_max,b1_Calf_max,
                                        b1_Hip_max,b1_Thigh_max,b1_Calf_max}; // upper command limit (robot space)

    const float lcommandlimit[12] = {   b1_Hip_min,b1_Thigh_min,b1_Calf_min,
                                        b1_Hip_min,b1_Thigh_min,b1_Calf_min,
                                        b1_Hip_min,b1_Thigh_min,b1_Calf_min,
                                        b1_Hip_min,b1_Thigh_min,b1_Calf_min}; // lower command limit (robot space)

    // conversion function
    float control2robotspace(float inp, int motorid);
    float robot2controlspace(float inp, int motorid);
    float interp(float x0, float x1, float t);
    float clamp(float x, float minn, float maxx);

    // Default standing pose in robot space (measured from real B1).
    // control space: hip=0.17, thigh=0.23, calf=-0.38 (same all legs)
    // Overwritten by control_initialized on first RobotControl() call.
    float motorcommand[12]    = {-0.03f, 1.08f, -1.94f,   // FR
                                  0.03f, 1.08f, -1.94f,   // FL
                                 -0.03f, 1.08f, -1.94f,   // RR
                                  0.03f, 1.08f, -1.94f};  // RL
    float oldmotorcommand[12] = {-0.03f, 1.08f, -1.94f,
                                  0.03f, 1.08f, -1.94f,
                                 -0.03f, 1.08f, -1.94f,
                                  0.03f, 1.08f, -1.94f};
    float interpmotorcommand[12] = {0}; // interpolated motor command
    float interpvelocitycommand[12] = {0}; // interpolated motor velocity command
    float position_error[12] = {0}; // position error
    float velocity_error[12] = {0}; // velocity error
    float control_input[12] = {0}; // control torque

    bool cmd_limit_warned[12] = {false}; // latch: command-limit warning already printed (per motor)
    bool fb_limit_warned[12] = {false};  // latch: feedback-limit warning already printed (per motor)

};




#endif
