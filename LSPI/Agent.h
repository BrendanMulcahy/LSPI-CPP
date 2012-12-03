#ifndef AGENT_H
#define AGENT_H

#include "constants.h"

class Agent
{
	public:
		int getAction(double x, double v)
		{
			int choice = rand() % 3;
			switch(choice)
			{
				case 0:
					return NF_OPT;
				case 1:
					return LF_OPT;
				case 2:
					return RF_OPT;
			}
		}

		void notify(double reward)
		{
		}
};

#endif