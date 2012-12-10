/**
 * Provides a wrapper class for matrix and vector operations.
 */

#include <cublas_v2.h>

class MatrixOps
{
	public:
		typedef struct matrix;
		typedef struct vector;

		static int errorCode;

		static void initializeCUDA();

		/**
		 * Returns a pointer to a vector of doubles, all set to 0.0
		 */
		static vector vec_zeros(int size);

		/**
		 * Returns a pointer representing a matrix square identity matrix.
		 */
		static matrix mat_eye(int size);

		/**
		 * Returns a Rows by Columns matrix of zeros.
		 */
		static matrix mat_zeroes(int rows, int columns);

		/**
		 * Retreives the element at the specified index
		 */
		static double mat_get(matrix mat, int row, int col);

		/**
		 * Sets the element at the specified index
		 */
		static void mat_set(matrix mat, int row, int col, double val);

		/**
		 * Retreives the element at the specified index
		 */
		static double vec_get(vector vec, int x);

		/**
		 * Sets the element at the specified index
		 */
		static void vec_set(vector vec, int x, double val);

		/**
		 * Computes the result of multiplying AxB.
		 */
		static matrix mult(matrix mat_A, matrix mat_B);

		/**
		 * Computes the result of the vector-matrix operation of mat * vec.
		 */
		static vector mult_vec(matrix mat, vector vec);

		/**
		 * Computes the result of the vector-matrix operation of vec * mat
		 */
		static vector mult_vec(vector vec, matrix mat);

		/**
		 * Calculates and returns the magnitude of the given vector.
		 * This is calculated by taking the  square root of the sum of squares for the vector components.
		 */
		static double mag_vec(vector vec);

		/**
		 * Frees the allocated memory from the matrix
		 */
		static void free_mat(matrix mat);

		/**
		 * Frees the allocated memory from the vector
		 */
		static void free_vec(vector matrix);

	private:
		static int devID;
		static cublasHandle_t handle;

		/**
		 * Sets the error code to -1 if the given (row, col) is out of bounds of mat.
		 * May have this print out an error.
		 */
		static void mat_in_bounds(matrix mat, int row, int col);

		/**
		 * Sets the error code to -1 if the given x is out of bounds of vec.
		 * May have this print out an error.
		 */
		static void vec_in_bounds(vector vec, int x);
};