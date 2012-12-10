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
int devID = 0;
cublasHandle_t handle;

typedef struct 
{
	double *matrix;
	int columns;
	int rows;
} matrix;

typedef struct 
{
	double *vector;
	int size;
} vector;

bool initializeCUDA()
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

bool initializeCUBLAS()
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

void destroy_context()
{
	cublasDestroy(handle);
}

/**
 * Sets the error code to -1 if the given (row, col) is out of bounds of mat.
 * May have this print out an error.
 */
bool mat_in_bounds(matrix mat, int row, int col)
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
bool vec_in_bounds(vector vec, int x)
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
vector vec_zeros(int size)
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
matrix mat_zeros(int rows, int columns)
{
	matrix mat;
	mat.columns = columns;
	mat.rows = rows;
	mat.matrix = (double *)malloc(sizeof(double)*mat.columns*mat.rows);
	
	return mat;
}

/**
 * Returns a pointer representing a matrix square identity matrix.
 */
matrix mat_eye(int size)
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
double mat_get(matrix mat, int row, int col)
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
void mat_set(matrix mat, int row, int col, double val)
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
double vec_get(vector vec, int x)
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
void vec_set(vector vec, int x, double val)
{
	if(!vec_in_bounds(vec, x))
	{
		return;
	}

	MatrixOps::errorCode = 0;
	vec.vector[x] = val;
}

bool exec_cudaMalloc(double *d_ptr, size_t mem_size)
{
	cudaError_t error;

	error = cudaMalloc(&d_ptr, mem_size);
	if(error != cudaSuccess)
	{
		printf("cudaMalloc return error code %d, line(%d)\n", error, __LINE__);
		return false;
	}

	return true;
}

bool exec_cudaMemcpy(double *dst, double *src, size_t size)
{
	cudaError_t error;

	error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
	{
		printf("cudaMemcpy return error code %d, line(%d)\n", error, __LINE__);
		return false;
	}

	return true;
}

bool exec_cublasSetMatrix(double *dst, matrix mat)
{
	cublasStatus_t status;

	status = cublasSetMatrix(mat.rows, mat.columns, sizeof(double), mat.matrix, , dst, );
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cudaSetMatrix return error code %d, line (%d)\n", status, __LINE__);
		return false;
	}

	return true;
}

bool exec_cublasDgemm(matrix mat_A, matrix mat_B, matrix mat_C)
{
	cublasStatus_t status;

	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_C.rows, mat_C.columns, mat_B.rows, 0.0, mat_
}

/**
 * Computes the result of multiplying AxB.
 * Sets the errorCode to -1 if an error is encountered.
 * Even if it errors you must call free.
 */
matrix mult(matrix mat_A, matrix mat_B)
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
	if(!(exec_cudaMalloc(dmat_A, size_A)
		&& exec_cudaMalloc(dmat_B, size_B)
		&& exec_cudaMalloc(dmat_result, size_result))
	  )
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	if(!(exec_cudaMemcpy(dmat_A, mat_A.matrix, size_A)
		&& exec_cudaMemcpy(dmat_B, mat_B.matrix, size_B)
		&& exec_cudaMemcpy(dmat_result, mat_result.matrix, size_result))
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

	if(!(exec_cublasDgemm(mat_A, mat_B, mat_result))
	{
		MatrixOps::errorCode = -1;
		return mat_result;
	}

	MatrixOps::errorCode = 0;
	return mat_result;
}

/**
 * Computes the result of the vector-matrix operation of mat * vec.
 * Sets the errorCode to -1 if an error is encountered.
 * Even if it errors you must call free.
 */
vector mult_vec(matrix mat, vector vec)
{
	vector vec_result = vec_zeros(mat.rows);

	if(mat.columns != vec.size)
	{
		printf("This won't work! Your dimensions suck: %dx%d * %d"
				, mat.rows
				, mat.columns
				, vec.size);
		MatrixOps::errorCode = -1;
		return vec_result;
	}

	// INSERT MULTIPLY CODE HERE

	MatrixOps::errorCode = 0;
	return vec_result;
}

/**
 * Computes the result of the vector-matrix operation of vec * mat.
 * Sets the errorCode to -1 if an error is encountered.
 * Even if it errors you must call free.
 */
vector mult_vec(vector vec, matrix mat)
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

	// INSERT MULTIPLY CODE HERE

	MatrixOps::errorCode = 0;
	return vec_result;
}

/**
 * Calculates and returns the magnitude of the given vector.
 * This is calculated by taking the  square root of the sum of squares for the vector components.
 */
double mag_vec(vector vec)
{
	double sum = 0.0;
	for(int i = 0; i < vec.size; i++)
	{
		sum += vec.vector[i]*vec.vector[i];
	}

	return std::sqrt(sum);
}