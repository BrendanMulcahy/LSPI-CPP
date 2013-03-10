#include "stdafx.h"
#include "Pendulum.h"
#include "constants.h"

/**
 * Constructor for objects of class Pendulum. Always starts with a random angle and angular velocity close to vertical.
 */
Pendulum::Pendulum()
{
	// Initalize to a random position and velocity within a small boundary.
	x = (float)(((float)rand()/(float)RAND_MAX)*((10.0f*M_PI)/180.0f) - ((5.0f*M_PI)/180.0f));
	v = (float)(((float)rand()/(float)RAND_MAX)*((10.0f*M_PI)/180.0f) - ((5.0f*M_PI)/180.0f));
}
/**
 * Returns true if the pendulum is horizontal, false otherwise.
 */
bool Pendulum::isHorizontal()
{
	if(abs(x) - M_PI/2.0 >= 0)
		return true;
	return false;
}

/**
 * Estimates the state change of the pendulum after the period of time, dt, has passed.
 * The action can be either NF_OPT, LF_OPT, or RF_OPT.
 */
void Pendulum::update(float dt, int action)
{
	float u;
	// figure out what force to use
	if(action == NF_OPT)
		u = NF_FORCE;
	else if(action == LF_OPT)
		u = LF_FORCE;
	else if(action == RF_OPT)
		u = RF_FORCE;
		
	// Check if we have hit 90 degrees, if so we are stable
	if(this->isHorizontal())
	{
		this->v = 0;
		if(this->x > 0)
			this->x = (float)(M_PI/2.0f);
		else
			this->x = (float)(-M_PI/2.0f);
	}
	else
	{
		u += (float)(((float)rand()/(float)RAND_MAX)*2.0f*(float)noise - noise); // Add noise to u
		float accel = (g_const*sin(this->x) - a_const*m_const*l_const*this->v*this->v*sin(2.0f*this->x)/2.0f - a_const*cos(this->x)*u);
		accel = accel/(4.0f*l_const/3.0f - a_const*m_const*l_const*pow(cos(this->x), 2));
		this->x += this->v*dt;
		this->v += accel*dt;
	}
}