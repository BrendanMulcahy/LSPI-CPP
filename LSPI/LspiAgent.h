#include "stdafx.h"
#include "Agent.h"
#include "armadillo"
#include <vector>
#include <array>

class LspiAgent: public Agent
{
	public:
		LspiAgent(std::vector<std::array<double, 7>> samples, double disc);

		int getAction(double x, double v);

	private:
		double discount;
		arma::vec w;
		
		// Each sample is (x, v, a, r, x', v')
		arma::vec lstdq(std::vector<std::array<double, 7>> samples);

		double magnitude(arma::vec vector);
	
		arma::vec basis_function(double x, double v, int action);
};