/**
 * Represents a "clever" agent which uses a simple static policy in an attempt to solve the inverted pendulum problem. 
 */

#include "stdafx.h"
#include "Agent.h"

class CleverAgent: public Agent
{
	public:
		/**
		 * Selects an action by separating the valid range of the pendulum into three sections: up, left, and right. If the pendulum is already upright, it will do nothing. 
		 * If the pendulum is leaning it will push depending on which direction the pendulum is leaning.
		 */
		int getAction(double x, double)
		{
			if(x < CUDART_PI_F/6.0 && x > -CUDART_PI_F/6.0)
				return NF_OPT;
			else if(x > CUDART_PI_F/6.0)
				return LF_OPT;
			else
				return RF_OPT;
		}
};