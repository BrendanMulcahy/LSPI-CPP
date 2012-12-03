#include "stdafx.h"
#include "Pendulum.h"
#include "constants.h"

Pendulum::Pendulum()
{
	// Initalize to a random position and velocity within a small boundary.
	x = ((double)rand()/(double)RAND_MAX)*((10.0*M_PI)/180.0) - ((5.0*M_PI)/180.0);
	v = ((double)rand()/(double)RAND_MAX)*((10.0*M_PI)/180.0) - ((5.0*M_PI)/180.0);
}

bool Pendulum::isHorizontal()
{
	if(abs(x) - M_PI/2.0 >= 0)
			return true;
	return false;
}

void Pendulum::update(double dt, double u)
{
	// figure out what force to use
	if(u == NF_OPT)
		u = NF_FORCE;
	else if(u == LF_OPT)
		u = LF_FORCE;
	else if(u == RF_OPT)
		u = RF_FORCE;
		
	// Check if we have hit 90 degrees, if so we are stable
	if(this->isHorizontal())
	{
		this->v = 0;
		if(this->x > 0)
			this->x = M_PI/2.0;
		else
			this->x = -M_PI/2.0;
	}
	else
	{
		u += ((double)rand()/(double)RAND_MAX)*2.0*(double)noise - noise; // Add noise to u
		double accel = (g_const*sin(this->x) - a_const*m_const*l_const*this->v*this->v*sin(2*this->x)/2.0 - a_const*cos(this->x)*u);
		accel = accel/(4.0*l_const/3.0 - a_const*m_const*l_const*pow(cos(this->x), 2));
		this->x += this->v*dt;
		this->v += accel*dt;
	}
}