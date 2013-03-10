#pragma once

struct sample
{
	float angle;
	float angular_velocity;
	int action;
	float reward;
	float final_angle;
	float final_angular_velocity;
	int terminal; 
};