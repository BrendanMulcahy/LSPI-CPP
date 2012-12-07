/**
 * Provides a wrapper class for matrix and vector operations.
 */

class MatrixOps
{
	public:
		typedef struct matrix;
		typedef struct vector;

		static int errorCode;

		/**
		 * Returns a pointer to a vector of doubles, all set to 0.0
		 */
		vector vec_zeros(int size);

		/**
		 * Returns a pointer representing a matrix square identity matrix.
		 */
		matrix mat_eye(int size);

		/**
		 * Returns a Rows by Columns matrix of zeros.
		 */
		matrix mat_zeroes(int rows, int columns);

		/**
		 * Retreives the element at the specified index
		 */
		double mat_get(matrix mat, int row, int col);

		/**
		 * Sets the element at the specified index
		 */
		void mat_set(matrix mat, int row, int col, double val);

		/**
		 * Retreives the element at the specified index
		 */
		double vec_get(vector vec, int x);

		/**
		 * Sets the element at the specified index
		 */
		void vec_set(vector vec, int x, double val);

		/**
		 * Computes the result of multiplying AxB.
		 */
		matrix mult(matrix mat_A, matrix mat_B);

		/**
		 * Computes the result of the vector-matrix operation of mat * vec.
		 */
		vector mult_vec(matrix mat, vector vec);

		/**
		 * Computes the result of the vector-matrix operation of vec * mat
		 */
		vector mult_vec(vector vec, matrix mat);

		/**
		 * Calculates and returns the magnitude of the given vector.
		 * This is calculated by taking the  square root of the sum of squares for the vector components.
		 */
		double mag_vec(vector vec);

		/**
		 * Frees the allocated memory from the matrix
		 */
		void free_mat(matrix mat);

		/**
		 * Frees the allocated memory from the vector
		 */
		void free_vec(vector matrix);

	private:
		/**
		 * Sets the error code to -1 if the given (row, col) is out of bounds of mat.
		 * May have this print out an error.
		 */
		void mat_in_bounds(matrix mat, int row, int col);

		/**
		 * Sets the error code to -1 if the given x is out of bounds of vec.
		 * May have this print out an error.
		 */
		void vec_in_bounds(vector vec, int x);
};