/**
 * Provides a wrapper class for matrix and vector operations.
 * Stores matrices in column major order to support CUBLAS.
 */

#include "stdafx.h"
#include "MatrixOps.h"
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

int MatrixOps::errorCode = 0;
int MatrixOps::devID = 0;
cublasHandle_t MatrixOps::handle;

extern "C" dgemm_(char* TRANSA, char* TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

/**
* Initializes the CUDA device.
*/
bool MatrixOps::initializeCUDA()
{
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;

    error = cudaSetDevice(devID);

    if (error != cudaSuccess)
    {
        printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
        return false;
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        return false;
    }

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        return false;
	}

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	return true;
}

/**
* Initializes the cublas handle.
*/
bool MatrixOps::initializeCUBLAS()
{
	cublasStatus_t ret;

	ret = cublasCreate(&handle);

	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
		return false;
	}

	return true;
}

/**
* Cleans up the CUDA/CUBLAS context. Must be called to free resources after initializeCUDA/CUBLAS.
*/
void MatrixOps::destroy_context()
{
	cublasDestroy(handle);
}

/**
 * Sets the error code to -1 if the given (row, col) is out of bounds of mat.
 * May have this print out an error.
 */
bool MatrixOps::mat_in_bounds(matrix mat, int row, int col)
{
	if(row >= mat.rows || col >= mat.columns)
	{
		printf("Out of bounds! (%d, %d) supplied, (%d, %d) available.\n", row, col, mat.rows, mat.columns);
		return false;
	}

	return true;
}

/**
* Sets the error code to -1 if the given x is out of bounds of vec.
*/
bool MatrixOps::vec_in_bounds(vector vec, int x)
{
	if(x >= vec.size)
	{
		printf("Out of bounds! %d supplied, %d available.\n", x, vec.size);
		MatrixOps::errorCode = -1;
		return false;
	}

	return true;
}

/**
 * Returns a pointer to a vector of doubles, all set to 0.0
 */
MatrixOps::vector MatrixOps::vec_zeros(int size)
{
	vector vec;
	vec.size = size;
	vec.vector = (double *)malloc(sizeof(double)*vec.size);
	for(int i = 0; i < vec.size; i++)
	{
		vec.vector[i] = 0.0;
	}

	return vec;
}

/**
* Returns a Rows by Columns matrix of zeros.
*/
MatrixOps::matrix MatrixOps::mat_zeros(int rows, int columns)
{
	matrix mat;
	mat.columns = columns;
	mat.rows = rows;
	mat.matrix = (double *)malloc(sizeof(double)*mat.columns*mat.rows);

	for(int col = 0; col < mat.columns; col++)
	{
		for(int row = 0; row < mat.rows; row++)
		{
			mat.matrix[col*mat.rows + row] = 0.0;
		}
	}
	
	return mat;
}

/**
 * Returns a pointer representing a matrix square identity matrix.
 */
MatrixOps::matrix MatrixOps::mat_eye(int size)
{
	matrix mat = mat_zeros(size, size);
	for(int col = 0; col < mat.columns; col++)
	{
		for(int row = 0; row < mat.rows; row++)
		{
			if(row == col)
				mat.matrix[col*mat.rows + row] = 1.0;
		}
	}

	return mat;
}

/**
 * Retreives the element at the specified index
 *
 * Sets the errorCode to -1 if the item is out of range
 */
double MatrixOps::mat_get(matrix mat, int row, int col)
{
	if(!mat_in_bounds(mat, row, col))
	{
		return false;
	}

	MatrixOps::errorCode = 0;
	return mat.matrix[col*mat.rows + row];
}

/**
 * Sets the element at the specified index
 *
 * Sets the errorCode to -1 if the item is out of range
 */
void MatrixOps::mat_set(matrix mat, int row, int col, double val)
{
	if(!mat_in_bounds(mat, row, col))
	{
		return;
	}

	MatrixOps::errorCode = 0;
	mat.matrix[col*mat.rows + row] = val;
}

/**
 * Retreives the element at the specified index
 *
 * Sets the errorCode to -1 if the item is out of range
 */
double MatrixOps::vec_get(vector vec, int x)
{
	if(!vec_in_bounds(vec, x))
	{
		return false;
	}

	MatrixOps::errorCode = 0;
	return vec.vector[x];
}

/**
 * Sets the element at the specified index
 *
 * Sets the errorCode to -1 if the item is out of range
 */
void MatrixOps::vec_set(vector vec, int x, double val)
{
	if(!vec_in_bounds(vec, x))
	{
		return;
	}

	MatrixOps::errorCode = 0;
	vec.vector[x] = val;
}

/**
* Wraps cuda malloc. Detects errors and reports true or false based on the result.
*/
bool MatrixOps::exec_cudaMalloc(double **d_ptr, size_t mem_size)
{
	cudaError_t error;

	error = cudaMalloc(d_ptr, mem_size);
	if(error != cudaSuccess)
	{
		printf("cudaMalloc return error code %d, line(%d)\n", error, __LINE__);
		return false;
	}

	return true;
}

/**
* Wraps cublas set matrix. Detects errors and reports true or false based on the result.
*/
bool MatrixOps::exec_cublasSetMatrix(double *dst, matrix mat)
{
	cublasStatus_t status;

	status = cublasSetMatrix(mat.rows, mat.columns, sizeof(double), mat.matrix, mat.rows, dst, mat.rows);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasSetMatrix return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
* Wraps cublas D gemm. Detects errors and reports true or false based on the result.
*/
bool MatrixOps::exec_cublasDgemm(double *dmat_A, double *dmat_B, double *dmat_C, int m, int k, int n)
{
	cublasStatus_t status;

	double alpha = 1.0;
	double beta = 0.0;
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
						 dmat_A, m, dmat_B, k, &beta, dmat_C, m);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDgemm return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
* Wraps cublas get matrix. Detects errors and reports true or false based on the result.
*/
bool MatrixOps::exec_cublasGetMatrix(double *src, matrix mat)
{
	cublasStatus_t status;

	status = cublasGetMatrix(mat.rows, mat.columns, sizeof(double), src, mat.rows, mat.matrix, mat.rows);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetMatrix return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas set vector.
 */
bool MatrixOps::exec_cublasSetVector(double *dvec, vector vec)
{
	cublasStatus_t status;

	status = cublasSetVector(vec.size, sizeof(double), vec.vector, 1, dvec, 1);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		if(status == CUBLAS_STATUS_INVALID_VALUE) { printf("\nInvalid value\n"); }
		if(status == CUBLAS_STATUS_MAPPING_ERROR) { printf("\nMapping error\n"); }
		printf("cublasSetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas get vector.
 */
bool MatrixOps::exec_cublasGetVector(double *dvec, vector vec)
{
	cublasStatus_t status;

	status = cublasGetVector(vec.size, sizeof(double), dvec, 1, vec.vector, 1);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas d scal
 */
bool MatrixOps::exec_cublasDscal(double alpha, double *dvec, int n)
{
	cublasStatus_t status;

	status = cublasDscal(handle, n, &alpha, dvec, 1);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas d gemv
 */
bool MatrixOps::exec_cublasDgemv(double *dvec, double *dmat, double *dvec_result, int m, int k, int n)
{
	cublasStatus_t status;

	cublasOperation_t trans;
	if(k == n)
	{
		trans = CUBLAS_OP_N;
	}
	else
	{
		trans = CUBLAS_OP_T;
	}
	
	double alpha = 1.0;
	double beta = 0.0;
	status = cublasDgemv(handle, trans, m, k, &alpha, dmat, m, dvec, 1, &beta, dvec_result, 1); 
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas d dot
 */
bool MatrixOps::exec_cublasDdot(double *dvec_A, double *dvec_B, double *result, int size)
{
	cublasStatus_t status;

	status = cublasDdot(handle, size, dvec_A, 1, dvec_B, 1, result);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Wrapper for cublas d axpy
 */
bool MatrixOps::exec_cublasDgeam(double *dmat_A, double *dmat_B, double *dmat_result, int m, int n, double beta)
{
	cublasStatus_t status;

	double alpha = 1.0;
	status = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, dmat_A, m, &beta, dmat_B, m, dmat_result, m);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasGetVector return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

/**
 * Computes the result of multiplying AxB.
 * Sets the errorCode to -1 if an error is encountered.
 * Even if it errors you must call free.
 */
MatrixOps::matrix MatrixOps::mult(matrix mat_A, matrix mat_B)
{
	matrix mat_result = mat_zeros(mat_A.rows, mat_B.columns);

	if(mat_A.columns != mat_B.rows)
	{
		printf("This won't work! Your dimensions suck: %dx%d * %dx%d"
				, mat_A.rows
				, mat_A.columns
				, mat_B.rows
				, mat_B.columns);
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	size_t size_A = sizeof(double)*mat_A.columns*mat_A.rows;
	size_t size_B = sizeof(double)*mat_B.columns*mat_B.rows;
	size_t size_result = sizeof(double)*mat_result.columns*mat_result.rows;

	// Allocate device memory
	double *dmat_A, *dmat_B, *dmat_result;
	if(!(exec_cudaMalloc(&dmat_A, size_A)
		&& exec_cudaMalloc(&dmat_B, size_B)
		&& exec_cudaMalloc(&dmat_result, size_result))
	  )
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// Copy from host to device
	if(!(exec_cublasSetMatrix(dmat_A, mat_A)
		&& exec_cublasSetMatrix(dmat_B, mat_B))
	  )
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// MULTIPLY
	if(!(exec_cublasDgemm(dmat_A, dmat_B, dmat_result, mat_A.rows, mat_A.columns, mat_B.columns)))
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// Retrieve the results
	if(!(exec_cublasGetMatrix(dmat_result, mat_result)))
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	cudaFree(dmat_A);
	cudaFree(dmat_B);
	cudaFree(dmat_result);

	MatrixOps::errorCode = 0;
	return mat_result;
}

/**
 * Multiplies in place every element of mat by alpha
 */
void MatrixOps::mult_in_place(double beta, matrix mat)
{
	double *dmat;
	size_t size = sizeof(double)*mat.rows*mat.columns;
	if(!(exec_cudaMalloc(&dmat, size)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// Copy from host to device
	if(!(exec_cublasSetMatrix(dmat, mat)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	cublasStatus_t status;

	double alpha = 0.0;
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mat.rows, mat.columns, mat.rows, &alpha,
						 dmat, mat.rows, dmat, mat.columns, &beta, dmat, mat.rows);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDgemm return error code %d, line (%d)\n", status, __LINE__);
		return;
	}

	// Retrieve the results
	if(!(exec_cublasGetMatrix(dmat, mat)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	cudaFree(dmat);

	MatrixOps::errorCode = 0;
	return;
}

/**
* Computes A*B
*/
double MatrixOps::dot(vector vec_A, vector vec_B)
{
	double *dvec_A, *dvec_B, result = 0.0;
	
	if(!exec_cudaMalloc(&dvec_A, sizeof(double)*vec_A.size))
	{
		MatrixOps::errorCode = -1;
		return result;
	}

	if(!exec_cudaMalloc(&dvec_B, sizeof(double)*vec_B.size))
	{
		MatrixOps::errorCode = -1;
		return result;
	}

	if(!exec_cublasSetVector(dvec_A, vec_A))
	{
		MatrixOps::errorCode = -1;
		return result;
	}

	if(!exec_cublasSetVector(dvec_B, vec_B))
	{
		MatrixOps::errorCode = -1;
		return result;
	} 

	if(!exec_cublasDdot(dvec_A, dvec_B, &result, vec_A.size))
	{
		MatrixOps::errorCode = -1;
		return result;
	}

	cudaFree(dvec_A);
	cudaFree(dvec_B);

	MatrixOps::errorCode = 0;
	return result;
}

/**
* computes A*B^T
*/
MatrixOps::matrix MatrixOps::mult(vector vec_A, vector vec_B)
{
	matrix mat_result = mat_zeros(vec_A.size, vec_B.size);

	if(vec_A.size != vec_B.size)
	{
		printf("This won't work! Your dimensions suck: %dx1 * 1%d"
				, vec_A.size
				, vec_B.size);
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// Allocate device memory
	double *dmat_A, *dmat_B, *dmat_result;
	if(!(exec_cudaMalloc(&dmat_A, sizeof(double)*vec_A.size)
		&& exec_cudaMalloc(&dmat_B, sizeof(double)*vec_B.size)
		&& exec_cudaMalloc(&dmat_result, sizeof(double)*vec_A.size*vec_B.size))
	  )
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// Copy from host to device
	if(!(exec_cublasSetVector(dmat_A, vec_A)
		&& exec_cublasSetVector(dmat_B, vec_B))
	  )
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// MULTIPLY
	if(!(exec_cublasDgemm(dmat_A, dmat_B, dmat_result, vec_A.size, 1, vec_B.size)))
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	// Retrieve the results
	if(!(exec_cublasGetMatrix(dmat_result, mat_result)))
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	cudaFree(dmat_A);
	cudaFree(dmat_B);
	cudaFree(dmat_result);

	MatrixOps::errorCode = 0;
	return mat_result;
}

/**
* Computes A + mod*B
*/
void MatrixOps::add(vector vec_A, vector vec_B, vector vec_result, double mod)
{
	if(vec_A.size != vec_B.size)
	{
		printf("This won't work! Your dimensions suck: %d + %d"
				, vec_A.size
				, vec_B.size);
		MatrixOps::errorCode = -1;
		return;
	}

	// Allocate device memory
	double *dvec_A, *dvec_B, *dvec_result;
	if(!(exec_cudaMalloc(&dvec_A, sizeof(double)*vec_A.size)
		&& exec_cudaMalloc(&dvec_B, sizeof(double)*vec_B.size)
		&& exec_cudaMalloc(&dvec_result, sizeof(double)*vec_A.size*vec_B.size))
	  )
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// Copy from host to device
	if(!(exec_cublasSetVector(dvec_A, vec_A)
		&& exec_cublasSetVector(dvec_B, vec_B))
	  )
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// MULTIPLY
	if(!(exec_cublasDgeam(dvec_A, dvec_B, dvec_result, vec_A.size, 1, mod)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// Retrieve the results
	if(!(exec_cublasGetVector(dvec_result, vec_result)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	cudaFree(dvec_A);
	cudaFree(dvec_B);
	cudaFree(dvec_result);

	MatrixOps::errorCode = 0;
	return;
}

/**
* Computes A + mod*B
*/
void MatrixOps::add(matrix mat_A, matrix mat_B, matrix mat_result, double mod)
{
	if(mat_A.rows != mat_B.rows || mat_A.columns != mat_B.columns)
	{
		printf("This won't work! Your dimensions suck: %dx%d * %dx%d"
				, mat_A.rows
				, mat_A.columns
				, mat_B.rows
				, mat_B.columns);
		MatrixOps::errorCode = -1;
		return;
	}

	// Allocate device memory
	double *dmat_A, *dmat_B, *dmat_result;
	if(!(exec_cudaMalloc(&dmat_A, sizeof(double)*mat_A.rows*mat_A.columns)
		&& exec_cudaMalloc(&dmat_B, sizeof(double)*mat_B.rows*mat_B.columns)
		&& exec_cudaMalloc(&dmat_result, sizeof(double)*mat_A.rows*mat_A.columns))
	  )
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// Copy from host to device
	if(!(exec_cublasSetMatrix(dmat_A, mat_A)
		&& exec_cublasSetMatrix(dmat_B, mat_B))
	  )
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// MULTIPLY
	if(!(exec_cublasDgeam(dmat_A, dmat_B, dmat_result, mat_A.rows, mat_A.columns, mod)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	// Retrieve the results
	if(!(exec_cublasGetMatrix(dmat_result, mat_result)))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	cudaFree(dmat_A);
	cudaFree(dmat_B);
	cudaFree(dmat_result);

	MatrixOps::errorCode = 0;
	return;
}

/**
 * Computes the result of the vector-matrix operation of mat * vec
 * Sets the errorCode to -1 if an error is encountered.
 * Even if it errors you must call free.
 */
MatrixOps::vector MatrixOps::mult_vec(vector vec, matrix mat)
{
	vector vec_result = vec_zeros(mat.columns);

	if(mat.columns != vec.size)
	{
		printf("This won't work! Your dimensions suck: %dx%d * %d"
				, mat.rows
				, mat.columns
				, vec.size);
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	double *dvec, *dmat, *dvec_result;
	if(!exec_cudaMalloc(&dvec, sizeof(double)*vec.size))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cudaMalloc(&dmat, sizeof(double)*mat.rows*mat.columns))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	size_t result_size;
	if(mat.rows == vec.size && mat.columns == vec.size)
	{
		result_size = mat.rows;
	}
	else if(mat.rows == vec.size && mat.columns != vec.size)
	{
		result_size = mat.columns;
	}
	else if(mat.rows != vec.size && mat.columns == vec.size)
	{
		result_size = mat.rows;
	}
	else
	{
		printf("Bad dimensions dummy. Your matrix is %dx%d and your vector is size %d", mat.rows, mat.columns, vec.size);
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cudaMalloc(&dvec_result, sizeof(double)*result_size))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cublasSetVector(dvec, vec))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cublasSetMatrix(dmat, mat))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cublasDgemv(dvec, dmat, dvec_result, mat.rows, mat.columns, result_size))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	if(!exec_cublasGetVector(dvec_result, vec_result))
	{
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	cudaFree(dvec);
	cudaFree(dmat);
	cudaFree(dvec_result);

	MatrixOps::errorCode = 0;
	return vec_result;
}

/**
 * Computes the result of the double-vector operation of alpha * vec and stores the result in vec.
 */
void MatrixOps::mult_vec_in_place(double alpha, vector vec)
{
	double *dvec;
	
	if(!exec_cudaMalloc(&dvec, sizeof(double)*vec.size))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	if(!exec_cublasSetVector(dvec, vec))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	if(!exec_cublasDscal(alpha, dvec, vec.size))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	if(!exec_cublasGetVector(dvec, vec))
	{
		MatrixOps::errorCode = -1;
		return;
	}

	cudaFree(dvec);

	MatrixOps::errorCode = 0;
}

/**
 * Calculates and returns the magnitude of the given vector.
 * This is calculated by taking the  square root of the sum of squares for the vector components.
 */
double MatrixOps::mag_vec(vector vec)
{
	double sum = 0.0;
	for(int i = 0; i < vec.size; i++)
	{
		sum += vec.vector[i]*vec.vector[i];
	}

	return std::sqrt(sum);
}

/**
* Calculates the magnitude of the difference between two vectors. Assumes both are the same size.
* Calculates vec_a - vec_b.
*/
double MatrixOps::mag_diff_vec(vector vec_a, vector vec_b)
{
	double sum = 0.0;
	for(int i = 0; i < vec_a.size; i++)
	{
		double temp = vec_a.vector[i] - vec_b.vector[i];
		sum += temp*temp;
	}

	return std::sqrt(sum);
}

/**
* Prints the contents of the matrix (rows -> columns).
*/
void MatrixOps::mat_print(matrix mat)
{
	for(int row = 0; row < mat.rows; row++)
	{
		for(int col = 0; col < mat.columns; col++)
		{
			printf("%f ", mat.matrix[col*mat.rows + row]);
		}
		printf("\n");
	}
}

/**
* Prints the contents of the vector.
*/
void MatrixOps::vec_print(vector vec)
{
	for(int i = 0; i < vec.size; i++)
	{
		printf("%f ", vec.vector[i]);
	}
	printf("\n");
}

/**
 *
 */
void MatrixOps::free_mat(matrix mat)
{
	free(mat.matrix);
}

/**
 *
 */
void MatrixOps::free_vec(vector vec)
{
	free(vec.vector);
}