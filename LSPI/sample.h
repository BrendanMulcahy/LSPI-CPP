#pragma once

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