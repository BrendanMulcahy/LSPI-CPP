/**
 * Represents an inverted pendulum.
 */

#pragma once

class Pendulum
{
	public:
		float x, v;

		/**
		 * Constructor for objects of class Pendulum. Always starts with a random angle and angular velocity close to vertical.
		 */
		Pendulum();

		/**
		 * Returns true if the pendulum is horizontal, false otherwise.
		 */
		bool isHorizontal();

		/**
		 * Estimates the state change of the pendulum after the period of time, dt, has passed.
		 * The action can be either NF_OPT, LF_OPT, or RF_OPT.
		 */
		void update(float dt, int action);
};