/**
 * Provides a wrapper class for matrix and vector operations.
 */

#include "stdafx.h"
#include "MatrixOps.h"
#include <cstdlib>
#include <cmath>

int MatrixOps::errorCode = 0;

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
	for(int row = 0; row < mat.rows; row++)
	{
		for(int col = 0; col < mat.columns; col++)
		{
			if(row == col)
				mat.matrix[row*mat.columns + col] = 1.0;
			else
				mat.matrix[row*mat.columns + col] = 0.0;
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
	return mat.matrix[row*mat.columns + col];
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
	mat.matrix[row*mat.columns + col] = val;
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

	// INSERT MULTIPLY CODE HERE

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