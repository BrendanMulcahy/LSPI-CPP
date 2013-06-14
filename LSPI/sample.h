#pragma once

#ifdef PENDULUM
struct sample
{
	float angle;
	float angular_velocity;
	int action;
	int reward;
	float final_angle;
	float final_angular_velocity;
	int terminal; 
};
#else

#include "QuakeDefs.h"

struct sample
{
	lspi_action_basis_t *state;
	lspi_action_basis_t *final_state;
	int action;
};

#endif