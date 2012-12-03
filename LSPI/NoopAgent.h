/**
 * Represents an agent which always chooses inaction. Provides a baseline for comparison against other bots.
 */

#include "stdafx.h"
#include "Agent.h"

class NoopAgent: public Agent
{
	public:
		/**
		 * Always returns NF_OPT
		 */
		int getAction(double x, double v)
		{
			return NF_OPT;
		}
};