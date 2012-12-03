#include "stdafx.h"
#include "Agent.h"

class CleverAgent: public Agent
{
	public:
		int getAction(double x, double v)
		{
			if(x < M_PI/6.0 && x > -M_PI/6.0)
				return NF_OPT;
			else if(x > M_PI/6.0)
				return LF_OPT;
			else
				return RF_OPT;
			return NF_OPT;
		}
};