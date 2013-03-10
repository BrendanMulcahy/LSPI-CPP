#include "stdafx.h"
#include "TestBlas.h"
#include "blas.h"
#include "Matrix.h"
#include <cmath>
#include <cublas_v2.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

#define Threshold 0.01f

bool TestBlas::run_tests()
{
	printf("\nInitializing CUDA & Cublas\n");
	cublasStatus_t stat = cublasCreate(&blas::handle);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("CUBLAS Init Failure.");
		return false;
	}

	printf("\nTesting Device Matrix.\n");

	// Verify we can get a zero matrix, should be hand checked
	printf("\nCreating a matrix of zeros of size 2x2:\n");
	Matrix<device_vector<float>> dmat0(2);
	thrust::fill(dmat0.vector.begin(), dmat0.vector.end(), 0.0);
	dmat0.print();
	
	//// Verify we can set values, should be hand
	printf("\nSet(0,0) = 1, (0, 1) = 2, (1, 0) = 3, (1, 1) = 4:\n");
	dmat0.set(0, 0, 1.0);
	dmat0.set(0, 1, 2.0);
	dmat0.set(1, 0, 3.0);
	dmat0.set(1, 1, 4.0);
	dmat0.print();

	printf("\nGet(0,0) (0, 1) (1, 0) (1, 1):\n");
	printf("%.1f ", dmat0.get(0,0));
	printf("%.1f", dmat0.get(0,1));
	printf("\n%.1f ", dmat0.get(1, 0));
	printf("%.1f\n", dmat0.get(1,1));

	printf("\nTesting Device Matrix-Matrix operations.\n");

	printf("\ngemm(A, B, C, alpha, beta):\n");
	Matrix<device_vector<float>> dmat1(2);
	dmat1.set(0, 0, 2.0);
	dmat1.set(0, 1, 0.0);
	dmat1.set(1, 0, 0.0);
	dmat1.set(1, 1, 2.0);

	Matrix<device_vector<float>> dmat2(2);
	dmat2.set(0, 0, 16.0);
	dmat2.set(0, 1, 12.0);
	dmat2.set(1, 0, 8.0);
	dmat2.set(1, 1, 4.0);

	float alpha = 2.0;
	float beta = 0.5;
	
	printf("%.1f\n*\n", alpha);
	dmat0.print();
	printf("*\n");
	dmat1.print();
	printf("+\n");
	printf("%.1f\n*\n", beta);
	dmat2.print();
	printf("=\n");
	
	if(blas::gemm(dmat0, dmat1, dmat2, alpha, beta) != 0)
	{
		printf("Failure to execute device gemm.");
		return false;
	}

	dmat2.print();

	if(!(abs(dmat2.get(0, 0) - 12.0) < Threshold &&
	   abs(dmat2.get(0, 1) - 14.0) < Threshold &&
	   abs(dmat2.get(1, 0) - 16.0) < Threshold &&
	   abs(dmat2.get(1, 1) - 18.0) < Threshold)
	  )
	{
		printf("Executed gemm but received an incorrect result.");
		return false;
	}

	printf("\ngeam(A, B, C, alpha, beta):\n");
	printf("%.1f\n*\n", alpha);
	dmat0.print();
	printf("+\n");
	printf("%.1f\n*\n", beta);
	dmat1.print();
	printf("=\n");

	if(blas::geam(dmat0, dmat1, dmat2, alpha, beta) != 0)
	{
		printf("Failure to execute device geam.");
		return false;
	}

	dmat2.print();

	if(!(abs(dmat2.get(0, 0) - 3.0) < Threshold &&
		abs(dmat2.get(0, 1) - 4.0) < Threshold &&
		abs(dmat2.get(1, 0) - 6.0) < Threshold &&
		abs(dmat2.get(1, 1) - 9.0) < Threshold)
	  )
	{
		printf("Executed geam but received an incorrect result.");
		return false;
	}

	printf("\nTesting Device Matrix-Vector operations.\n");

	printf("\ngemv(A, x, y, alpha, beta, false):\n");
	device_vector<float> dvec0(2);
	dvec0[0] = -1.0;
	dvec0[1] = 0.0;

	device_vector<float> dvec1(2);
	dvec1[0] = 1.0;
	dvec1[1] = 1.5;

	printf("%.1f\n*\n", alpha);
	dmat0.print();
	printf("*\n%.1f %.1f\n+\n", (float)dvec0[0], (float)dvec0[1]);
	printf("%.1f\n*\n%.1f %.1f\n=\n", beta, (float)dvec1[0], (float)dvec1[1]);

	if(blas::gemv(dmat0, dvec0, dvec1, alpha, beta, false) != 0)
	{
		printf("Failure to execute device gemv.");
		return false;
	}

	printf("%.1f %.1f\n", (float)dvec1[0], (float)dvec1[1]);

	if(!(abs(dvec1[0] + 1.5) < Threshold &&
		abs(dvec1[1] + 5.25) < Threshold)
	  )
	{
		printf("Executed gemv but received an incorrect result.");
		return false;
	}

	printf("\ngemv(A, x, true):\n");
	printf("%.1f %.1f\n*\n", (float)dvec0[0], (float)dvec0[1]);
	dmat0.print();
	printf("=\n");

	if(blas::gemv(dmat0, dvec0, dvec1, true) != 0)
	{
		printf("Failed to execute gemv.");
		return false;
	}

	printf("%.1f %.1f\n", (float)dvec1[0], (float)dvec1[1]);

	if(!(abs(dvec1[0] + 1.0) < Threshold &&
		abs(dvec1[1] + 2.0) < Threshold)
	  )
	{
		printf("Executed gemv but received an incorrect result.");
		return false;
	}

	printf("\nger(x, y, A, alpha):\n");
	printf("%.1f %.1f\nX\n%.1f %.1f\n+\n", (float)dvec0[0], (float)dvec0[1], (float)dvec1[0], (float)dvec1[1]);
	dmat0.print();
	printf("=\n");
	
	if(blas::ger(dvec0, dvec1, dmat0, alpha))
	{
		printf("Failed to execute ger.");
		return false;
	}

	dmat0.print();

	if(!(abs(dmat0.get(0, 0) - 3.0) < Threshold &&
		abs(dmat0.get(0, 1) - 6.0) < Threshold &&
		abs(dmat0.get(1, 0) - 3.0) < Threshold &&
		abs(dmat0.get(1, 1) - 4.0) < Threshold)
	  )
	{
		printf("Executed ger but received an incorrect result.");
		return false;
	}

	printf("\nTesting Device Vector-Vector operations.\n");

	printf("\nscal(x, alpha):\n");
	printf("%.1f\n*\n%.1f %.1f\n=\n", alpha, (float)dvec0[0], (float)dvec0[1]);

	if(blas::scal(dvec0, alpha) != 0)
	{
		printf("Failed to execute scal.");
		return false;
	}

	printf("%.1f %.1f\n", (float)dvec0[0], (float)dvec0[1]);

	if(!(abs(dvec0[0] + 2.0) < Threshold &&
		abs(dvec0[1] - 0.0) < Threshold)
	  )
	{
		printf("Executed scal but received an incorrect result.");
		return false;
	}

	printf("\ndot(x, y, result):\n");
	printf("%.1f %.1f\ndot\n%.1f %.1f\n=\n", (float)dvec0[0], (float)dvec0[1], (float)dvec1[0], (float)dvec1[1]);

	alpha = 0.0;
	if(blas::dot(dvec0, dvec1, alpha) != 0)
	{
		printf("Failed to execute dot.");
		return false;
	}

	printf("%.1f\n", alpha);

	if(!(abs(alpha - 2.0) < Threshold))
	{
		printf("Executed dot but received an incorrect result.");
		return false;
	}

	printf("\naxpy(x, y, alpha):\n");
	printf("%.1f\n*\n%.1f %.1f\n+\n%.1f %.1f\n=\n", alpha, (float)dvec0[0], (float)dvec0[1], (float)dvec1[0], (float)dvec1[1]);

	if(blas::axpy(dvec0, dvec1, alpha) != 0)
	{
		printf("Failed to execute axpy.");
		return false;
	}

	printf("%.1f %.1f\n", (float)dvec1[0], (float)dvec1[1]);

	if(!(abs(dvec1[0] + 5.0) < Threshold &&
		abs(dvec1[1] + 2.0) < Threshold)
	  )
	{
		printf("Executed axpy but received an incorrect result.");
		return false;
	}

	printf("\nPassed!\n");
	return true;
}