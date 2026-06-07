/*****************************************************************
								myb1
******************************************************************/

/* -----------------------------------------------------------------
							import libraries
------------------------------------------------------------------ */


// unitree datatype
#include "b1_interface/myb1.h"

/* -----------------------------------------------------------------
						standard unitree protocol
------------------------------------------------------------------ */

void myB1::UDPRecv()
{
	udp.Recv();
}

void myB1::UDPSend()
{
	udp.Send();
}

/* -----------------------------------------------------------------
						    	initialization function
------------------------------------------------------------------ */

/* -----------------------------------------------------------------
							set function
------------------------------------------------------------------ */

void myB1::setMotorCommand(float newmotorcommand[])
{
	for (int i=0;i<12;i++)
	{
		oldmotorcommand[i] = motorcommand[i];
		motorcommand[i] = control2robotspace(newmotorcommand[i],i);
	}
	ti = dt*rate;
	intp_t = 0;
	// An explicit command was received — mark control as initialized so the
	// seed-from-current-state block in RobotControl() does not overwrite it.
	control_initialized = true;
}

void myB1::setPD(PIDgains* pidgains)
{
	bool gains_changed = false;
	for (int i=0;i<12;i++)
	{
		float newkp, newkd, newtau;
		if (i%3 == 0)
		{
			newkp = (float)pidgains->hip.kp;
			newkd = (float)pidgains->hip.kd;
			newtau = pidgains->hip.tau;
		}
		else if (i%3 == 1)
		{
			newkp = (float)pidgains->thigh.kp;
			newkd = (float)pidgains->thigh.kd;
			newtau = pidgains->thigh.tau;
		}
		else
		{
			newkp = (float)pidgains->knee.kp;
			newkd = (float)pidgains->knee.kd;
			newtau = pidgains->knee.tau;
		}

		if ((kp[i] != newkp) || (kd[i] != newkd) || (tau[i] != newtau)) gains_changed = true;

		kp[i] = newkp;
		kd[i] = newkd;
		tau[i] = newtau;
	}

	// Print gains only when they actually change (startup, or after `ros2 param set`)
	// instead of every 50 Hz tick — keeps full debug visibility without flooding stdout.
	if (gains_changed)
	{
		cout << "[INFO]: PD gains updated (idx  kp  kd  tau):" << endl;
		for (int i=0;i<12;i++)
			cout << "  [" << i << "]\t" << kp[i] << "\t" << kd[i] << "\t" << tau[i] << endl;
	}
}

/* -----------------------------------------------------------------
							get function
------------------------------------------------------------------ */

float myB1::getMotorPosition(int id)
{
	return robot2controlspace(state.motorState[id].q,id);
}

/* -----------------------------------------------------------------
					   conversion functions
------------------------------------------------------------------ */

float myB1::control2robotspace(float inp, int motorid)
{
	float op = direction[motorid]*inp + offset[motorid];

	if ((op > ucommandlimit[motorid]) || (op < lcommandlimit[motorid])){
		if (!cmd_limit_warned[motorid]) { // warn once per excursion, not on every call
			cout << "\033[1;33m[WARNING]: command exceeds joint limits.\033[0m" << endl;
			cout << "[INFO]: motor command " << motorid << " is " << op << " rads, which exceeds [" << lcommandlimit[motorid] << "," << ucommandlimit[motorid] << "]." << endl;
			cmd_limit_warned[motorid] = true;
		}
	} else {
		cmd_limit_warned[motorid] = false;
	}
	return clamp(op,lcommandlimit[motorid],ucommandlimit[motorid]);
}

float myB1::robot2controlspace(float inp, int motorid)
{
	float op = (inp - offset[motorid])/direction[motorid];

	if ((inp > ucommandlimit[motorid]) || (inp < lcommandlimit[motorid])){
		if (!fb_limit_warned[motorid]) { // warn once per excursion, not on every call (50 Hz)
			cout << "\033[1;33m[WARNING]: feedback exceeds joint limits.\033[0m" << endl;
			cout << "[INFO]: motor " << motorid << " is at " << inp << " rads, which exceeds [" << lcommandlimit[motorid] << "," << ucommandlimit[motorid] << "]." << endl;
			fb_limit_warned[motorid] = true;
		}
	} else {
		fb_limit_warned[motorid] = false;
	}
	return op;
}

float myB1::interp(float x0, float x1, float t)
{
	float t_ = clamp(t,0.0,1.0);
	return x0*(1-t_) + x1*t_;
}

float myB1::clamp(float x, float minn, float maxx)
{
	if (minn <= maxx){
		return std::min(std::max(x,minn),maxx);
	}else{
		return std::min(std::max(x,maxx),minn);
	}

}

/* -----------------------------------------------------------------
							control loop
------------------------------------------------------------------ */

void myB1::RobotControl()
{
	udp.GetRecv(state);

	// On first run, seed motorcommand from the current joint positions so the
	// controller starts with zero error instead of trying to drive every joint
	// to raw-angle-zero (which is not the standing pose).
	if (!control_initialized)
	{
		for (int i = 0; i < 12; i++)
		{
			motorcommand[i]    = state.motorState[i].q; // robot space
			oldmotorcommand[i] = state.motorState[i].q;
		}
		control_initialized = true;
		cout << "[INFO]: RobotControl initialized from current joint positions." << endl;
	}

	if (t < 1)
	{
		t += (DT/TMAX);
		// Print once per second (every 1/DT ticks = 1000 ticks at 1 kHz)
		int ticks = (int)(t / DT);
		if (ticks % 1000 == 0)
			cout << "[INFO]: soft start " << (int)(t*100) << " %." << endl;
	}else{
		if (t != 1) cout << "[INFO]: soft start 100 % — full gains active." << endl;
		t = 1;
	}

	intp_t += (dt);



	for (int i=0;i<12;i++)
	{

		// trajectory generation/interpolation
		interpmotorcommand[i] = interp(oldmotorcommand[i], motorcommand[i],  intp_t*rate);
		interpvelocitycommand[i] = 1*(motorcommand[i]-oldmotorcommand[i])*rate;

		// error
		position_error[i] = interpmotorcommand[i] - state.motorState[i].q;
		velocity_error[i] = 0*interpvelocitycommand[i] - state.motorState[i].dq;



		/*
		// mussel model (adaptive Kp, Kd, tau) : comment out if not needed

		bool cond = (i == 2)||(i == 5)||(i == 8)||(i == 11)

		if (cond)
		{
			velocity_error[i] = interpvelocitycommand[i] - state.motorState[i].dq;
		}else{
			velocity_error[i] = 0*interpvelocitycommand[i] - state.motorState[i].dq;
		}


		if (t >= 1) // mussel model (adaptive Kp, Kd, tau) : comment out if not needed
		{

			if (cond)
			{

				float a = 0.15;     //35.0, 0.2
  	    		float b = 50.0;     //5.0
  	    		float beta =  0.0;

  	    		float pos_diff = position_error[i];
				float vel_diff = velocity_error[i];

				float tra_diff = position_error[i] + beta * vel_diff;
			    float co_diff = a / (1.00 + b * tra_diff * tra_diff);

			    float ff = tra_diff / co_diff;

			    float kp_in = ff * pos_diff;
			    float kd_in = ff * vel_diff;
			    float t_in  = ff;


			    kp[i] = kp_in;
			    kd[i] = kd_in;
			    tau[i] = t_in;
			}
		} */



		// compute control input (PID)
		// All three terms are ramped by t so the robot starts passive (t=0) and
		// builds up stiffness, damping, and gravity-comp together. Without ramping
		// tau, the hip feedforward (-4 Nm) applied at t=0 with zero stiffness
		// causes the hip to drift before the restoring force exists.
		control_input[i] = t * (kp[i]*position_error[i] + kd[i]*velocity_error[i] + tau[i]);
		control_input[i] = clamp(control_input[i],-tau_max,tau_max);
	}

	//cout << control_input[0] <<  " " << control_input[1] << " "<< control_input[2] << endl;
	/* -----------------------------------------------------------------
				comment this section to disable the motor
	------------------------------------------------------------------ */

	
	// ----------------------   set robot command
	for (int i=0;i<12;i++)
	{
		cmd.motorCmd[i].q = interpmotorcommand[i];
		cmd.motorCmd[i].dq = interpvelocitycommand[i];				//0*(motorcommand[1]-oldmotorcommand[1])*rate;
		cmd.motorCmd[i].Kp = 0;										//(int)(t*kp[i]);
		cmd.motorCmd[i].Kd = 0;										//kd[i];
		cmd.motorCmd[i].tau = control_input[i];
	}

	safe.PositionLimit(cmd);

	udp.SetSend(cmd);
	



	/* -----------------------------------------------------------------
				comment this section to disable the motor
	------------------------------------------------------------------ */



}


/* -----------------------------------------------------------------
							safety function
------------------------------------------------------------------ */

void myB1::cutoff(int status)
{
	if (safety_triggered) return; // already latched; ignore any further triggers

	cout << "\033[1;31m[ERROR]: safety violation shutdown due to ";
	switch (status)
	{
		case 0:
			cout << "the power protection.";
			break;
		case 1:
			cout << "the shoutdown command received.";
			break;
		case 2:
			cout << "the connection problem (/B1/connection topic).";
			break;
		case 3 ... 6:
			cout << "the joint limit protection at the " << legname[status-3] << " leg.";
			break;
		case 7 ... 18:
			cout << "the joint speed protection at the " << legname[(int)((status-7)/3)] << (int)((status-7)%3) << " joint.";
			break;
		case 19 ... 30:
			cout << "the joint overheat at the " << legname[(int)((status-19)/3)] << (int)((status-19)%3)  << " joint.";
			break;
		default:
			cout << "an unknown reason.";
			break;
	}
	cout << "\033[0m" << endl;

	// Graceful stop: command passive damping (sent only if motor_output is enabled),
	// then latch a safe state. Replaces the previous hard abort() so the threads can
	// shut down cleanly and the node keeps publishing telemetry.
	Damp();
	safety_triggered = true;
}

void myB1::SafetyCheck()
{
	if (safety_triggered) return; // latched safe state - stop monitoring until restart

	udp.GetRecv(state);
	memcpy(&jointStickCommand,&state.wirelessRemote[0],40);

	// ---------------------------------  unitree power protection (unknown definition)
	if (safe.PowerProtect(cmd,state,1) < 0) cutoff(0);

	// ---------------------------------  joystick cutoff
	// A / X / Y alone → hard kill (no modifier combination uses these as safe commands)
	if ((int)jointStickCommand.btn.components.A > 0) cutoff(1);
	if ((int)jointStickCommand.btn.components.X > 0) cutoff(1);
	if ((int)jointStickCommand.btn.components.Y > 0) cutoff(1);
	// B alone → hard kill.  L2+B → normal damping command (do NOT latch safety stop).
	bool L2_held = (int)jointStickCommand.btn.components.L2 > 0;
	if ((int)jointStickCommand.btn.components.B > 0 && !L2_held) cutoff(1);

	// ---------------------------------  joint limit protection
	for (int i=0;i<4;i++)
	{
		float motorposition_ = getMotorPosition(3*i+1);
		if ((motorposition_ > CUTOFF_UPPERLIM) or (motorposition_ < CUTOFF_LOWERLIM)) cutoff(i+3);
	}

	// --------------------------------- joint speed & temp protection
	for (int i=0;i<12;i++)
	{
		if (fabs(state.motorState[i].dq) > speed_max) cutoff(7+i); // speed cut off
		if (((int)state.motorState[i].temperature) > CUTOFF_TEMP) cutoff(19+i); // temp cut off
	}
}

void myB1::BatteryCheck()
{
	udp.GetRecv(state);

	if (((float)state.bms.SOC) < 10.0) { cout << "\033[1;33m[WARNING]: battery is below 10%.\033[0m" << endl;}

	for (int i=0;i<15;i++)
	{
		float vi = (float)state.bms.cell_vol[i]/1000;
		if (vi < 2.0) {cout << "\033[1;33m[WARNING]: low voltage at cell " << i << "\033[0m" << endl;}
	}
}

/* -----------------------------------------------------------------
					motor-enable reset helper
------------------------------------------------------------------ */

void myB1::resetForMotorEnable()
{
	// Re-seed commanded position from current measured state and reset the
	// soft-start ramp. Call this every time motor_output transitions false → true.
	//
	// Without this, enabling motor_output after a long idle period fires with:
	//   t = 1.0   → full gains immediately (no ramp)
	//   tau[hip] = -4 Nm feedforward → immediate hip jerk even with zero position error
	//
	udp.GetRecv(state);
	for (int i = 0; i < 12; i++)
	{
		motorcommand[i]    = state.motorState[i].q;
		oldmotorcommand[i] = state.motorState[i].q;
	}
	intp_t = 0;
	t      = 0;   // restart soft-start ramp: gains rise over TMAX seconds
	cout << "[INFO]: resetForMotorEnable — seeded from current pose, soft-start reset." << endl;
}

/* -----------------------------------------------------------------
						watchdog / damping helpers
------------------------------------------------------------------ */

void myB1::Damp()
{
	// Passive damping command: zero stiffness + light velocity damping.
	// q/dq use the Unitree "ignore" sentinels so the motor firmware applies pure
	// Kd damping (the robot sinks slowly instead of collapsing or going wild).
	for (int i=0;i<12;i++)
	{
		cmd.motorCmd[i].mode = 0x0A; // foc / servo mode
		cmd.motorCmd[i].q    = PosStopF;
		cmd.motorCmd[i].dq   = VelStopF;
		cmd.motorCmd[i].Kp   = 0.0f;
		cmd.motorCmd[i].Kd   = damp_kd;
		cmd.motorCmd[i].tau  = 0.0f;
	}

	// Always transmit the damping command — Damp() is a STOP signal, not a
	// motor-output signal. motor_output=false while not yet running = dry-run
	// (Damp is never called). motor_output=false while running = deliberate
	// stop — the robot must receive the damping command or it keeps executing
	// the last frozen torque and drifts.
	udp.SetSend(cmd);
}

void myB1::Hold()
{
	// Hold last position on a deliberate stop (motor_output=false, command stale, etc.).
	// Switches to onboard PD (Kp/Kd nonzero) targeting the current measured position,
	// so the robot stays put instead of sinking under gravity (Damp behavior).
	udp.GetRecv(state);
	for (int i=0;i<12;i++)
	{
		cmd.motorCmd[i].mode = 0x0A;
		cmd.motorCmd[i].q    = state.motorState[i].q; // hold actual current position
		cmd.motorCmd[i].dq   = 0.0f;
		cmd.motorCmd[i].Kp   = kp[i];
		cmd.motorCmd[i].Kd   = kd[i];
		cmd.motorCmd[i].tau  = tau[i]; // keep feedforward for gravity compensation
	}
	udp.SetSend(cmd);
}

unsigned long long myB1::getRecvCount()
{
	return udp.udpState.RecvCount;
}

bool myB1::safetyTriggered()
{
	return safety_triggered;
}
