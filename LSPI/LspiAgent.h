#pragma once

/**
 * Provides an implemenation of the Least Squared Policy Iteration algorithm for solving the inverted pendulum problem.
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

#define NUM_ACTIONS 3
#define BASIS_SIZE 4
#define SIGMA_2 1

//#define VERBOSE_HIGH
//#define VERBOSE_MED
//#define VERBOSE_LOW

#if defined(VERBOSE_HIGH)
#	define VERBOSE_MED
#	define VERBOSE_LOW
#elif defined(VERBOSE_MED)
#	define VERBOSE_LOW
#endif

#define PRINT(X) do															\
	{																		\
		printf("\n");														\
		for(int z = 0; z < X.size(); z++) { printf("%.6f\n", (float)X[z]); } \
		printf("\n");														\
    } while(0)

using namespace thrust;
using namespace blas;

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
		LspiAgent(thrust::host_vector<sample> samples, float disc) : discount(disc), w(BASIS_SIZE*NUM_ACTIONS)
		{
			thrust::fill(w.begin(), w.end(), 0.0f);

			// Loop until policy converges
			vector_type policy = lstdq(samples);
#if defined(VERBOSE_HIGH)
			PRINT(policy);
#endif
			vector_type temp(policy);
#if defined(VERBOSE_HIGH)
			PRINT(w);
#endif
			blas::axpy(w, temp, -1.0f);
#if defined(VERBOSE_HIGH)
			PRINT(temp);
#endif

			//TODO: Write a magnitude function dammit!
			float magnitude = 0.0f;
			for(int i = 0; i < temp.size(); i++)
			{
				magnitude += temp[i]*temp[i];
			}


			int k = 0;
			while(sqrt(magnitude) > epsilon_const && k < 10)
			{
				w = policy;
#if defined(VERBOSE_LOW)
				PRINT(w);
#endif
				policy = lstdq(samples);

				vector_type temp2(policy);
				blas::axpy(w, temp2, -1.0f);
				//TODO: Write a magnitude function dammit!
				magnitude = 0.0f;
				for(int i = 0; i < temp2.size(); i++)
				{
					magnitude += temp2[i]*temp2[i];
				}
				k++;
			}

#if defined(VERBOSE_LOW)
			PRINT(policy);
#endif

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
			int options[3] = {NF_OPT, LF_OPT, RF_OPT};
			for(int i = 0; i < 3; i++)
			{
				vector_type params = basis_function(x, v, options[i]);
				float q; 
				dot(params, w, q);
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
		
		// TODO: Test the speed penalty of copying samples to the GPU for gpu based implementation
		/**
		 * Given a set of samples, performs a single update step on the current agent's policy.
		 */
		vector_type lstdq(thrust::host_vector<sample> samples)
		{
			Matrix<vector_type> B(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(B.vector.begin(), B.vector.end(), 0.0f);

			// TODO: Put this in a function for both vector_types and write a custom CUDA kernel for the GPU implementation
			for(int i = 0; i < B.rows; i++)
			{
				for(int j = 0; j < B.rows; j++)
				{
					if(i == j)
						B.set(i, j, 1.0f);
				}
			}

			scal(B.vector, 0.1f); // TODO: Investigate the importance and effect of this number
			
#if defined(VERBOSE_HIGH)
			printf("\n");
			B.print();
#endif

			vector_type b(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(b.begin(), b.end(), 0.0f);

			for(unsigned int i = 0; i < samples.size(); i++)
			{
				// Get the basis functions
				vector_type phi = basis_function(samples[i].angle, samples[i].angular_velocity, (int)(samples[i].action));
				int next_action = getAction(samples[i].final_angle, samples[i].final_angular_velocity);
				vector_type phi_prime = basis_function(samples[i].final_angle, samples[i].final_angular_velocity, next_action);

				// Break the calculation into smaller parts
				scal(phi_prime, discount);
				axpy(phi, phi_prime, -1.0f); // TODO: Consider optimizing this by creating a custom kernel
				scal(phi_prime, -1.0f); // This is because axpy does not allow us to do y = x - y, only y = y - x
				
#if defined(VERBOSE_HIGH)
				printf("\n");
				B.print();
				printf("\n");
				PRINT(phi);
				PRINT(phi_prime);
#endif

				// TODO: Try to eliminate extra memory allocation by reusing vectors
				vector_type temp(phi.size());
				vector_type temp2(phi.size());
				Matrix<vector_type> num(BASIS_SIZE*NUM_ACTIONS);
				thrust::fill(num.vector.begin(), num.vector.end(), 0.0f);

				gemv(B, phi, temp, false);
				gemv(B, phi_prime, temp2, true);
				ger(temp, temp2, num);
				
#if defined(VERBOSE_HIGH)
				PRINT(temp);
				PRINT(temp2);
				num.print();
#endif

				float denom;
				dot(phi, temp2, denom);
				denom += 1.0f;
				
#if defined(VERBOSE_HIGH)
				printf("\n%.3f\n", denom);
#endif

				scal(num.vector, 1.0f/denom);
				axpy(num.vector, B.vector, -1.0f);
				
#if defined(VERBOSE_HIGH)
				num.print();
				B.print();
#endif

				// Update values
				scal(phi, (float)samples[i].reward);
				axpy(phi, b);
				
#if defined(VERBOSE_HIGH)
				printf("\n%d\n", samples[i].reward);

				PRINT(phi);
				PRINT(b);
#endif
			}

#if defined(VERBOSE_MED)
			B.print();
			PRINT(b);
#endif
			vector_type temp_b(b.size());
			gemv(B, b, temp_b, false);
			b = temp_b;
			
#if defined(VERBOSE_MED)
			PRINT(b);
#endif

			return b;
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

