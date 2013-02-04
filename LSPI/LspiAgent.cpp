#include "stdafx.h"
#include "LspiAgent.h"
#include <thrust\device_vector.h>

using namespace std;
using namespace thrust;
		
/**
 * Given a set of samples, performs a single update step on the current agent's policy.
 */
template <>
host_vector<float> LspiAgent<host_vector<float>>::lstdq(host_vector<float[7]> samples)
{
	MatrixOps::matrix B = MatrixOps::mat_eye(BASIS_SIZE*NUM_ACTIONS);
	MatrixOps::mult_in_place(0.1, B);
	MatrixOps::vector b = MatrixOps::vec_zeros(BASIS_SIZE*NUM_ACTIONS);
	for(int i = 0; i < samples.size(); i++)
	{
		// Get the basis functions
		host_vector<float> phi = basis_function(samples[i][0], samples[i][1], samples[i][2]);
		int next_action = getAction(samples[i][4], samples[i][5]);
		host_vector<float> phi_prime = basis_function(samples[i][4], samples[i][5], next_action);

		// Break the calculation into smaller parts
		MatrixOps::mult_vec_in_place(discount, phi_prime);
		MatrixOps::add(phi, phi_prime, phi_prime, -1.0);
		MatrixOps::vector temp = MatrixOps::mult_vec(phi, B);
		MatrixOps::vector temp2 = MatrixOps::mult_vec(phi_prime, B);
		MatrixOps::matrix num = MatrixOps::mult(temp, temp2);

		double denom = 1.0 + MatrixOps::dot(temp2, phi);
		MatrixOps::mult_in_place(1.0/denom, num);
		MatrixOps::add(B, num, B, -1.0);

		// Update values
		MatrixOps::mult_vec_in_place(samples[i][3], phi);
		MatrixOps::add(b, phi, b, 1.0);

		MatrixOps::free_vec(phi);
		MatrixOps::free_vec(phi_prime);
		MatrixOps::free_vec(temp);
		MatrixOps::free_vec(temp2);
		MatrixOps::free_mat(num);
	}
	
	MatrixOps::vector result = MatrixOps::mult_vec(b, B);

	MatrixOps::free_mat(B);
	MatrixOps::free_vec(b);

	return result;
}

/**
 * Returns the policy function weights for the given angle, velocity, and action.
 * These weights can be used to compute the estimated fitness of the given action.
 */
template<>
host_vector<float> LspiAgent<host_vector<float>>::basis_function(float x, float v, int action)
{
	MatrixOps::vector phi = MatrixOps::vec_zeros(BASIS_SIZE*NUM_ACTIONS);

	// If we're horizontal then the basis function is all 0s
	if (fabs(x) - M_PI/(2.0) >= 0)
	{
		return phi;
	}

	// Now populate the basis function for this state action pair
	// Note that each entry except for the first is a gaussian.
	int i = BASIS_SIZE * (action);
	MatrixOps::vec_set(phi, i, 1.0);
	i += 1;
	float value = M_PI/2.0;
	for(float j = -value; j <= value; j += (value/((BASIS_SIZE-1)/6)))
	{
		for(float k = -1; k <= 1; k += 1)
		{
			float dist = (x - j)*(x - j) + (v - k)*(v - k);
			MatrixOps::vec_set(phi, i, exp(-dist/(2*SIGMA_2)));
			i += 1;
		}
	}

	return phi;
}

// Explicit template instantiation
template class LspiAgent<host_vector<float>>;
template class LspiAgent<host_vector<float>>;