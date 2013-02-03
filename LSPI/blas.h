#include "Matrix.h"

namespace blas
{
	/**
	 * Computes C = alpha*A*B + beta*C
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemm(const Matrix<vec_type>& A, const Matrix<vec_type>& B, Matrix<vec_type>& C, float alpha, float beta);

	/**
	 * Computes C = alpha*A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemm(const Matrix<vec_type>& A, const Matrix<vec_type>& B, Matrix<vec_type>& C, float alpha);

	/**
	 * Computes C = A*B
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemm(const Matrix<vec_type>& A, const Matrix<vec_type>& B, Matrix<vec_type>& C);

	/**
	 * Computes A = alpha*A
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemm(Matrix<vec_type>& A, float alpha);

	/**
	 * Computes x = alpha*x. 
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int scal(vec_type<vec_type>& x, float alpha); 

	/**
	 * Computes result = x dot y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int dot(const vec_type& x, const vec_type& y, float& result);
	
	/**
	 * Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemv(const Matrix<vec_type>& A, const vec_type& x, vec_type& y, float alpha, float beta, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemv(const Matri<vec_type>x& A, const vec_type& x, vec_type& y, float alpha, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set tranpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int gemv(const Matrix<vec_type>& A, const vec_type& x, vec_type& y, bool transpose);

	/**
	 * Computes C = alpha*A + beta*B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int geam(const Matrix<vec_type>& A, const Matrix<vec_type>& B, Matrix<vec_type>& C, float alpha, float beta);

	/**
	 * Computes C = A + B.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	template <class vec_type>
	int geam(const Matrix<vec_type>& A, const Matrix<vec_type>& B, Matrix & C);
};