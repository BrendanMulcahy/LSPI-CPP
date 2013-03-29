/**
 * Provides a wrapper around cblas and some custom blas-like extension functions for computation on the CPU.
 */

#include "stdafx.h"
#include "blas.h"
#include "cblas.h"

using namespace thrust;

static int inc = 1;

#define PRINT(X) do															\
	{																		\
		printf("\n");														\
		for(int z = 0; z < X.size(); z++) { printf("%.6f\n", (float)X[z]); } \
		printf("\n");														\
    } while(0)

/**
* Computes C = alpha*A*B + beta*C
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta)
{
	char trans = NORMAL;
	sgemm_(&trans, &trans, &C.rows, &C.cols, &A.cols, &alpha, raw_pointer_cast(A.vector.data()), &A.rows, raw_pointer_cast(B.vector.data()), &B.rows,
		   &beta, raw_pointer_cast(C.vector.data()), &C.rows);
	return 0;
}

/**
* Computes C = alpha*A*B
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha)
{
	blas::gemm(A, B, C, alpha, 0.0);
	return 0;
}

/**
* Computes C = A*B
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C)
{
	blas::gemm(A, B, C, 1.0, 0.0);
	return 0;
}

/**
* Computes A = alpha*A
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(Matrix<host_vector<float>>& A, float alpha)
{
	blas::gemm(A, A, A, 0.0, alpha);
	return 0;
}

/**
* Computes x = alpha*x. 
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::scal(host_vector<float>& x, float alpha)
{
	int m = (int)x.size();
	sscal_(&m, &alpha, raw_pointer_cast(x.data()), &inc);
	return 0;
}

/**
* Computes result = x dot y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::dot(const host_vector<float>& x, const host_vector<float>& y, float& result)
{
	int m = (int)x.size();
	result = sdot_(&m, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc);
	return 0;
}
	
/**
* Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, float beta, bool transpose)
{
	char trans;
	if(transpose)
	{
		trans = TRANSPOSE;
	}
	else
	{
		trans = NORMAL;
	}

	sgemv_(&trans, &A.rows, &A.cols, &alpha, raw_pointer_cast(A.vector.data()), &A.rows, raw_pointer_cast(x.data()), &inc, &beta, 
		   raw_pointer_cast(y.data()), &inc);
	return 0;
}

/**
* Computes y = alpha*A*x. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, bool transpose)
{
	return blas::gemv(A, x, y, alpha, 0.0, transpose);
}

/**
* Computes y = A*x. For x*A set tranpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, bool transpose)
{
	return blas::gemv(A, x, y, 1.0, 0.0, transpose);
}

/**
* Computes C = alpha*A + beta*B.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::geam(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta)
{
	// This is a bit slower than the GPU implementation since it relies on AXPY behind the scenes. Should not significantly impact the results
	// since we are more concerned with scale and expecting perf increases in the range 100x. At best this could be completed about 3x as fast
	// (algorithmically) and only represents part of the LSPI implementation.
	if(&A == &C)
	{
		int rval = blas::scal(C.vector, alpha);
		if(rval != 0) { return rval; }

		return blas::axpy(B.vector, C.vector, beta);
	}
	else if(&B == &C)
	{
		int rval = blas::scal(C.vector, beta);
		if(rval != 0) { return rval; }

		return blas::axpy(A.vector, C.vector, alpha);
	}

	// Make sure C is zero'd out since axpy will always use the value of both vectors
	thrust::fill(C.vector.begin(), C.vector.end(), 0.0f);

	int rval = blas::axpy(A.vector, C.vector, alpha);
	if(rval != 0) { return rval; }

	return blas::axpy(B.vector, C.vector, beta);
}

/**
* Computes C = A + B.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::geam(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C)
{
	return blas::geam(A, B, C, 1.0, 1.0);
}

/**
* Computes y = alpha*x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const host_vector<float>& x, host_vector<float>& y, float alpha)
{
	int m = (int)x.size();
	saxpy_(&m, &alpha, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc);
	return 0;
}

/**
* Computes y = x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const host_vector<float>& x, host_vector<float>& y)
{
	return blas::axpy(x, y, 1.0);
}

/**
* Computes A = alpha*x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A, float alpha)
{
	int m = (int)x.size();
	int n = (int)y.size();
	sger_(&m, &n, &alpha, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc, raw_pointer_cast(A.vector.data()), &A.rows);
	return 0;
}

/**
* Computes A = x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A)
{
	blas::ger(x, y, A, 1.0);
	return 0;
}