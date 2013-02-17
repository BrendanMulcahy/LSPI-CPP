/**
 * Provides a wrapper around cblas and some custom blas-like extension functions for computation on the CPU.
 */

#include "stdafx.h"
#include "blas.h"
#include "cblas.h"

/**
* Computes C = alpha*A*B + beta*C
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta)
{
	return 0;
}

/**
* Computes C = alpha*A*B
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha)
{
	return blas::gemm(A, B, C, alpha, 0.0);
}

/**
* Computes C = A*B
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C)
{
	return blas::gemm(A, B, C, 1.0, 0.0);
}

/**
* Computes A = alpha*A
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemm(Matrix<host_vector<float>>& A, float alpha)
{
	return blas::gemm(A, A, A, 0.0, 1.0);
}

/**
* Computes x = alpha*x. 
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::scal(host_vector<float>& x, float alpha)
{
	return 0;
}

/**
* Computes result = x dot y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::dot(const host_vector<float>& x, const host_vector<float>& y, float& result)
{
	return 0;
}
	
/**
* Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, float beta, bool transpose)
{
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

// TODO: geam does not exist in BLAS, I need to write an actual host implementation
/**
* Computes C = alpha*A + beta*B.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::geam(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta)
{
	return 0;
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