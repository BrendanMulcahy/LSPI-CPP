/**
 * Provides an implemenation of the Least Squared Policy Iteration algorithm for solving the inverted pendulum problem.
 *
 * Two template implementations are provided and only these two implementations should be used:
 * *thrust::host_vector
 * *thrust::device_vector
 */

#include "stdafx.h"
#include "Agent.h"
#include <vector>
#include <array>
#include "MatrixOps.h"
#include <thrust\host_vector.h>

template <typename vector_type>
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
		LspiAgent(thrust::host_vector<float[7]> samples, float discount)
		{
			MatrixOps::initializeCUDA();
			MatrixOps::initializeCUBLAS();

			discount = disc;
			w = MatrixOps::vec_zeros(BASIS_SIZE*NUM_ACTIONS);

			// Loop until policy converges
			vector_type policy = lstdq(samples);
			while(MatrixOps::mag_diff_vec(w, policy) > epsilon_const)
			{
				w = policy;
				policy = lstdq(samples);
			}

			w = policy;
		}

		/**
		 * After creation, the LSPI Agent's policy is used to generate a functional value at a given angle and velocity. This functional value defines the action
		 * the agent intends to take.
		 */
		int getAction(float x, float v)
		{
			int action = -9999;
			float max = -9999;
			int options[3] = {RF_OPT, NF_OPT, LF_OPT};
			for(int i = 0; i < 3; i++)
			{
				vector_type params = basis_function(x, v, options[i]);
				float q = MatrixOps::dot(params, w);
				if(q > max)
				{
					action = options[i];
					max = q;
				}
			}

			return action;
		}

	private:
		float discount;
		vector_type w;
		
		/**
		 * Given a set of samples, performs a single update step on the current agent's policy.
		 */
		vector_type lstdq(thrust::host_vector<float[7]> samples);
	
		/**
		 * Returns the policy function weights for the given angle, velocity, and action.
		 * These weights can be used to compute the estimated fitness of the given action.
		 */
		vector_type basis_function(float x, float v, int action);
};

