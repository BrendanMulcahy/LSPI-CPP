/**
 * Provides a thin wrapper around thrust vectors to support matrix style operations.
 */

template <class vec_type> 
class Matrix
{
	public:
		/**
		 * Creates a matrix of size Rows by Cols. The cublas handle specified will be used
		 * to execute all GPU accelerated operations. The matrix has no values explicitly set.
		 */
		Matrix(int rows, int cols);

		/**
		 * Frees the memory allocated to contain the matrix.
		 */
		~Matrix();

		/**
		 * Sets the matrix to a matrix of zeros.
		 */
		void makeZeros();

		/**
		 * Sets the matrix to an identity matrix, such that for every coordinate (x, y), 
		 * get(x, y) = 1.0 iff x == y, otherwise get(x, y) = 0.0.
		 */ 
		void makeIdentity();

		/**
		 * Sets the value of coordinate (row, col) to val.
		 */
		void set(int row, int col, float val);

		/**
		 * Returns the value of coordinate (row, col).
		 */
		float get(int row, int col);

		/**
		 * Returns a string representation of the matrix (each row is on a new line).
		 */
		char* toString();
};