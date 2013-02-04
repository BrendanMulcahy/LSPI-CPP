#include "Matrix.h"
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

using namespace thrust;

namespace blas
{
	//********** HOST CALLS **********//

	/**
	 * Computes C = alpha*A*B + beta*C
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta);

	/**
	 * Computes C = alpha*A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha);

	/**
	 * Computes C = A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C);

	/**
	 * Computes A = alpha*A
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(Matrix<host_vector<float>>& A, float alpha);

	/**
	 * Computes x = alpha*x. 
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int scal(host_vector<float>& x, float alpha); 

	/**
	 * Computes result = x dot y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int dot(const host_vector<float>& x, const host_vector<float>& y, float& result);
	
	/**
	 * Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, float beta, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set tranpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, bool transpose);

	/**
	 * Computes C = alpha*A + beta*B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int geam(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C, float alpha, float beta);

	/**
	 * Computes C = A + B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int geam(const Matrix<host_vector<float>>& A, const Matrix<host_vector<float>>& B, Matrix<host_vector<float>>& C);

	//********** DEVICE CALLS **********//

	/**
	 * Computes C = alpha*A*B + beta*C
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<device_vector<float>>& A, const Matrix<device_vector<float>>& B, Matrix<device_vector<float>>& C, float alpha, float beta);

	/**
	 * Computes C = alpha*A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<device_vector<float>>& A, const Matrix<device_vector<float>>& B, Matrix<device_vector<float>>& C, float alpha);

	/**
	 * Computes C = A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(const Matrix<device_vector<float>>& A, const Matrix<device_vector<float>>& B, Matrix<device_vector<float>>& C);

	/**
	 * Computes A = alpha*A
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemm(Matrix<device_vector<float>>& A, float alpha);

	/**
	 * Computes x = alpha*x. 
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int scal(device_vector<float>& x, float alpha); 

	/**
	 * Computes result = x dot y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int dot(const device_vector<float>& x, const device_vector<float>& y, float& result);
	
	/**
	 * Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, float beta, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set tranpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, bool transpose);

	/**
	 * Computes C = alpha*A + beta*B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int geam(const Matrix<device_vector<float>>& A, const Matrix<device_vector<float>>& B, Matrix<device_vector<float>>& C, float alpha, float beta);

	/**
	 * Computes C = A + B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int geam(const Matrix<device_vector<float>>& A, const Matrix<device_vector<float>>& B, Matrix<device_vector<float>>& C);
};