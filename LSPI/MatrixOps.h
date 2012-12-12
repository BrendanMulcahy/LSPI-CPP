/**
 * Provides a wrapper class for matrix and vector operations.
 */

#include <cublas_v2.h>

class MatrixOps
{
	public:
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

		static int errorCode;

		/**
		 * Initializes the CUDA device. Must be called followed by initializeCUBLAS in order to use the multiply operations.
		 */
		static bool initializeCUDA();

		/**
		 * Initializes the cublas handle.
		 */
		static bool initializeCUBLAS();

		/**
		 * Cleans up the CUDA/CUBLAS context. Must be called to free resources after initializeCUDA/CUBLAS.
		 */
		static void destroy_context();

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
		static matrix mat_zeros(int rows, int columns);

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
		 * Calculates the magnitude of the difference between two vectors
		 */
		static double mag_diff_vec(vector vec_a, vector vec_b);

		/**
		 * Frees the allocated memory from the matrix
		 */
		static void free_mat(matrix mat);

		/**
		 * Frees the allocated memory from the vector
		 */
		static void free_vec(vector matrix);

		/**
		 * Prints the size contents of the matrix (rows -> columns).
		 */
		static void mat_print(matrix mat);

	private:
		static int devID;
		static cublasHandle_t handle;

		/**
		 * Sets the error code to -1 if the given (row, col) is out of bounds of mat.
		 * May have this print out an error.
		 */
		static bool mat_in_bounds(matrix mat, int row, int col);

		/**
		 * Sets the error code to -1 if the given x is out of bounds of vec.
		 * May have this print out an error.
		 */
		static bool vec_in_bounds(vector vec, int x);

		/**
		 * Wraps cuda malloc. Detects errors and reports true or false based on the result.
		 */
		static bool exec_cudaMalloc(double **dmat_A, size_t size_A);

		/**
		 * Wraps cublas set matrix. Detects errors and reports true or false based on the result.
		 */
		static bool exec_cublasSetMatrix(double *dmat_A, matrix mat_A);

		/**
		 * Wraps cublas D gemm. Detects errors and reports true or false based on the result.
		 */
		static bool exec_cublasDgemm(double *dmat_A, double *dmat_B, double *dmat_result, int m, int k, int n);

		/**
		 * Wraps cublas get matrix. Detects errors and reports true or false based on the result.
		 */
		static bool exec_cublasGetMatrix(double *dmat_result, matrix mat_result);
};