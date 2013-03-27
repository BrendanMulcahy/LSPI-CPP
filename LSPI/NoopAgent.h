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
		int getAction(double, double)
		{
			return NF_OPT;
		}
};