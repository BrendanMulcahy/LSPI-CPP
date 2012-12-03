/**
 * Provides an implemenation of the Least Squared Policy Iteration algorithm for solving the inverted pendulum problem.
 */

#include "stdafx.h"
#include "Agent.h"
#include "armadillo"
#include <vector>
#include <array>

class LspiAgent: public Agent
{
	public:
		/**
		 * To create an LSPI Agent, a discount factor and a large number of sample data points are required. More sample should result in a better policy.
		 * The samples should come from data taken from an agent performing at random, or a previous iteration of an LSPI Agent.
		 *
		 * Each sample in the vector should be of the format [x, v, a, r, x', v', t]
		 * -x is the angle
		 * -v is the angular velocity
		 * -a is action selected
		 * -r is the reward received after executing the action
		 * -x' is the angle after executing the action
		 * -v' is the angular velocity after executing the action
		 * -t is 1 if the state after executing is terminal, 0 otherwise
		 */
		LspiAgent(std::vector<std::array<double, 7>> samples, double disc);

		/**
		 * After creation, the LSPI Agent's policy is used to generate a functional value at a given angle and velocity. This functional value defines the action
		 * the agent intends to take.
		 */
		int getAction(double x, double v);

	private:
		double discount;
		arma::vec w;
		
		/**
		 * Given a set of samples, performs a single update step on the current agent's policy.
		 */
		arma::vec lstdq(std::vector<std::array<double, 7>> samples);

		/**
		 * Calculates and returns the magnitude of the given vector.
		 * This is calculated by taking the  square root of the sum of squares for the vector components.
		 */
		double magnitude(arma::vec vector);
	
		/**
		 * Returns the policy function weights for the given angle, velocity, and action.
		 * These weights can be used to compute the estimated fitness of the given action.
		 */
		arma::vec basis_function(double x, double v, int action);
};