#include "stdafx.h"
#include "Agent.h"

class NoopAgent: public Agent
{
	public:
		int getAction(double x, double v)
		{
			return NF_OPT;
		}
};