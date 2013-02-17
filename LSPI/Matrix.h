#pragma once

/**
 * Provides a thin wrapper around thrust vectors to support matrix style operations.
 */
template <class vec_type> 
class Matrix
{
	public:
		vec_type vector;
		int rows, cols;

		/**
		 * Creates a matrix of size NxN. The matrix has no values explicitly set.
		 */
		Matrix(int n)
		{
			rows = n;
			cols = n;
			vector(n*n);
		}

		/**
		 * Creates a matrix of size MxN. The matrix has no values explicitly set.
		 */
		Matrix(int m, int n)
		{
			rows = m;
			cols = n;
			vector(rows*cols);
		}

		// TODO: I don't think we actually need to do this
		///**
		// * Frees the memory allocated to contain the matrix.
		// */
		//~Matrix()
		//{
		//	vector.clear();

		//}

		/**
		 * Sets the value of coordinate (row, col) to val.
		 */
		void set(int row, int col, float val)
		{
			vector[col*rows + row] = val;
		}

		/**
		 * Returns the value of coordinate (row, col).
		 */
		float get(int row, int col)
		{
			return vector[col*rows + row];
		}

		// TODO: Fill this in for debugging
		/**
		 * Returns a string representation of the matrix (each row is on a new line).
		 */
	//	char* toString();
};