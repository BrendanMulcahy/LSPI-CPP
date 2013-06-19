#pragma once

/**
 * Provides an implementation of OLPOMDP (Baxter & Bartlett, 2000)
 *
 * Two template implementations are provided and only these two implementations should be used:
 * *thrust::host_vector
 * *thrust::device_vector
 */

#include "stdafx.h"
#include "sample.h"
#include "Agent.h"
#include <vector>
#include "blas.h"
#include "Matrix.h"
#include <thrust\host_vector.h>
#include <Windows.h>
#include <stdlib.h>
#include <cmath>

#define SIGMA_2 1
#define E_CONST 2.71828f

#define VERBOSE_HIGH

#define PRINT(X) do															 \
	{																		 \
		printf("\n");														 \
		for(int z = 0; z < X.size(); z++) { printf("%.6f\n", (float)X[z]); } \
		printf("\n");														 \
    } while(0)

using namespace thrust;
using namespace blas;

template <typename vector_type>
class GradientAgent: public Agent
{
	public:
		GradientAgent(thrust::host_vector<float> policy, float stepsize, float bias) : w(policy), 
			gamma(stepsize), beta(bias), eligibility(NUM_ACTIONS*BASIS_SIZE) 
		{
			thrust::fill(eligibility.begin(), eligibility.end(), 0.0f);
		}

		/**
		 * Updates the policy online given the sample s. This algorithm assumes that s was dervied
		 * from the initial state and an action specified by getAction(state).
		 */
		void update(sample s)
		{
			vector_type basis = basis_function(s.angle, s.angular_velocity, s.action);
			vector_type basis_prime(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(basis_prime.begin(), basis_prime.end(), 0.0f);
			
			vector_type basis_nf = basis_function(s.final_angle, s.final_angular_velocity, NF_OPT);
			vector_type basis_lf = basis_function(s.final_angle, s.final_angular_velocity, LF_OPT);
			vector_type basis_rf = basis_function(s.final_angle, s.final_angular_velocity, RF_OPT);

			// Calculate dem probabilities
			float nf = getPartialProbability(s.angle, s.angular_velocity, NF_OPT);
			float rf = getPartialProbability(s.angle, s.angular_velocity, RF_OPT);
			float lf = getPartialProbability(s.angle, s.angular_velocity, LF_OPT);
			float total = nf + rf + lf;
			nf = nf/total;
			lf = lf/total;
			rf = rf/total;

			// Now we compute basis_prime = nf*basis_nf + lf*basis_lf + rf*basis_rf
			axpy(basis_nf, basis_prime, nf);
			axpy(basis_lf, basis_prime, lf);
			axpy(basis_rf, basis_prime, rf);

			// The update step for e: e <- beta*e + basis - basis_prime
			PRINT(eligibility);
			scal(eligibility, beta);
			PRINT(eligibility);
			axpy(basis, eligibility);
			PRINT(eligibility);
			axpy(basis_prime, eligibility, -1.0);
			
			PRINT(eligibility);
			PRINT(w);
			// The update step for w: w <- w + gamma*reward*e
			vector_type step(eligibility);
			axpy(step, w, gamma*s.reward);
			PRINT(w);
		}

		/**
		 * After creation, the LSPI Agent's policy is used to generate a functional value at a given angle and velocity. This functional value defines the action
		 * the agent intends to take.
		 */
		int getAction(float x, float v)
		{
			float nf, lf, rf, total;

			nf = getPartialProbability(x, v, NF_OPT);
			lf = getPartialProbability(x, v, LF_OPT);
			rf = getPartialProbability(x, v, RF_OPT);

			// Divide the basis results by the total to determine the probability distribution
			total = nf + lf + rf;
			nf = nf/total;
			lf = lf/total;
			rf = rf/total;

			// Now select one of these options
			float select = (float)rand()/RAND_MAX;
			if(select < nf)
			{
				return NF_OPT;
			}

			select -= nf;
			if(select < lf)
			{
				return LF_OPT;
			}

			return RF_OPT;
		}

	private:
		float gamma, beta;
		vector_type w;
		vector_type eligibility;

		/**
		 * Calculates e^J_theta(s, a), the partial probability of the action for use in the boltzmann distribution.
		 */
		float getPartialProbability(float x, float v, int action)
		{
			float val;
			vector_type params = basis_function(x, v, action);
			dot(params, w, val);

			return pow(E_CONST, (float)(val/1e6));
		}
	
		/**
		 * Returns the policy function weights for the given angle, velocity, and action.
		 * These weights can be used to compute the estimated fitness of the given action.
		 */
		vector_type basis_function(float x, float v, int action)
		{
			vector_type phi(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(phi.begin(), phi.end(), 0.0f);
			
#if defined(VERBOSE_HIGH)
			PRINT(phi);
#endif

			// If we're horizontal then the basis function is all 0s
			if (fabs(x) - CUDART_PI_F/(2.0f) >= 0)
			{
				return phi;
			}

			// TODO: Move this into a transform/cuda kernel
			// Now populate the basis function for this state action pair
			// Note that each entry except for the first is a gaussian.
			int i = BASIS_SIZE * (action-1);
			phi[i] = 1.0f;
			i += 1;
			float value = CUDART_PI_F/4.0f;
			for(float j = -value; j <= value; j += (value/((BASIS_SIZE-1)/6.0) + 0.0001))
			{
				for(float k = -1; k <= 1; k += 1)
				{
					float dist = (x - j)*(x - j) + (v - k)*(v - k);
					phi[i] = exp(-dist/(2*SIGMA_2));
					i += 1;
				}
			}

#if defined(VERBOSE_HIGH)
			PRINT(phi);
#endif

			return phi;
		}
};

