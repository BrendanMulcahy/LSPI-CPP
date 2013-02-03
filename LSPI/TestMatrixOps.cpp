#include "stdafx.h"
#include "TestMatrixOps.h"
#include "MatrixOps.h"

#define CHECK_ERROR() \
	do { \
		if(MatrixOps::errorCode != 0) \
		{ \
			printf("Error occurred: {0}", MatrixOps::errorCode); \
			return false; \
		} \
	} \
	while(0)

#define VERIFY_VEC_GET(vec, index, value) \
	do { \
		int temp = MatrixOps::vec_get(vec, index); \
		CHECK_ERROR(); \
		if(temp != value) \
		{ \
			printf("Failed vec_get. Expected {0}, returned {1}\n.", value, temp); \
			return false; \
		} \
	} \
	while(0)

#define VERIFY_MAT_GET(mat, x, y, value) \
	do { \
		int temp = MatrixOps::mat_get(mat, x, y); \
		CHECK_ERROR(); \
		if(temp != value) \
		{ \
			printf("Failed vec_get. Expected {0}, returned {1}\n.", value, temp); \
			return false; \
		} \
	} \
	while(0)

bool TestMatrixOps::run_tests()
{
	printf("Initializing CUDA & Cublas\n");
	MatrixOps::initializeCUDA();
	CHECK_ERROR();
	MatrixOps::initializeCUBLAS();
	CHECK_ERROR();

	printf("Testing vectors.\n");

	// Verify we can get a zero vector, should be hand checked if vec_get fails.
	printf("\nvec_zeros(4):\n");
	MatrixOps::vector vec0 = MatrixOps::vec_zeros(4);
	CHECK_ERROR();
	MatrixOps::vec_print(vec0);
	
	// Verify we can set values, should be hand checked if vec_get fails.
	printf("\nvec_set(0, 1) (1, 2) (2, 3) (3, 4):\n");
	MatrixOps::vec_set(vec0, 0, 1);
	CHECK_ERROR();
	MatrixOps::vec_set(vec0, 1, 2);
	CHECK_ERROR();
	MatrixOps::vec_set(vec0, 2, 3);
	CHECK_ERROR();
	MatrixOps::vec_set(vec0, 3, 4);
	CHECK_ERROR();
	MatrixOps::vec_print(vec0);

	// Verify we can get values
	printf("\nvec_get(0) (1) (2) (3):\n");
	VERIFY_VEC_GET(vec0, 0, 1);
	VERIFY_VEC_GET(vec0, 1, 2);
	VERIFY_VEC_GET(vec0, 2, 3);
	VERIFY_VEC_GET(vec0, 3, 4);
	printf("Passed\n");

	printf("\nmag_vec():\n");
	{
		double mag = MatrixOps::mag_vec(vec0);
		if(mag != sqrt(30.0))
		{
			printf("Failed mag_vec. Returned %f, expected %f", mag, sqrt(30.0));
			return false;
		}
		printf("Passed\n");
	}

	printf("\nmult_vec_in_place(2.0):\n");
	MatrixOps::mult_vec_in_place(2.0, vec0);
	CHECK_ERROR();
	MatrixOps::vec_print(vec0);

	VERIFY_VEC_GET(vec0, 0, 2);
	VERIFY_VEC_GET(vec0, 1, 4);
	VERIFY_VEC_GET(vec0, 2, 6);
	VERIFY_VEC_GET(vec0, 3, 8);

	printf("\nCreating a second vector:\n");
	MatrixOps::vector vec1 = MatrixOps::vec_zeros(4);
	CHECK_ERROR();
	MatrixOps::vec_set(vec1, 0, 1);
	CHECK_ERROR();
	MatrixOps::vec_set(vec1, 1, 2);
	CHECK_ERROR();
	MatrixOps::vec_set(vec1, 2, 3);
	CHECK_ERROR();
	MatrixOps::vec_set(vec1, 3, 4);
	CHECK_ERROR();
	MatrixOps::vec_print(vec1);

	printf("\nmag_diff_vec():\n");
	{
		double mag = MatrixOps::mag_diff_vec(vec0, vec1);
		if(mag != sqrt(30.0))
		{
			printf("Failed mag_diff_vec. Returned %f, expected %f", mag, sqrt(30.0));
			return false;
		}
		printf("Passed\n");
	}

	printf("\nadd()\n");
	MatrixOps::vec_print(vec0);
	printf("+\n");
	MatrixOps::vec_print(vec1);
	printf("=\n");
	MatrixOps::add(vec0, vec1, vec1, 1.0);
	MatrixOps::vec_print(vec1);

	printf("\nadd(-1)\n");
	MatrixOps::vec_print(vec1);
	printf("-\n");
	MatrixOps::vec_print(vec0);
	printf("=\n");
	MatrixOps::add(vec1, vec0, vec1, -1.0);
	MatrixOps::vec_print(vec1);

	printf("\nmult()\n");
	MatrixOps::matrix mat = MatrixOps::mult(vec0, vec1);
	CHECK_ERROR();
	MatrixOps::vec_print(vec0);
	printf("*\n");
	MatrixOps::vec_print(vec1);
	printf("=\n");
	MatrixOps::mat_print(mat);

	printf("\nfree_vec() & free_mat()\n");
	MatrixOps::free_vec(vec0);
	CHECK_ERROR();
	MatrixOps::free_vec(vec1);
	CHECK_ERROR();
	MatrixOps::free_mat(mat);
	CHECK_ERROR();
	printf("Passed.\n");

	printf("\n\nTesting Matrices.\n");

	printf("\nmat_zeros(2, 2)\n");
	MatrixOps::matrix mat0 = MatrixOps::mat_zeros(2, 2);
	CHECK_ERROR();
	MatrixOps::mat_print(mat0);

	printf("\nmat_eye(2)\n");
	MatrixOps::matrix mat1 = MatrixOps::mat_eye(2);
	CHECK_ERROR();
	MatrixOps::mat_print(mat1);

	printf("\nmat_set()\n");
	MatrixOps::mat_set(mat0, 0, 0, 1.0);
	CHECK_ERROR();
	MatrixOps::mat_set(mat0, 0, 1, 2.0);
	CHECK_ERROR();
	MatrixOps::mat_set(mat0, 1, 0, 3.0);
	CHECK_ERROR();
	MatrixOps::mat_set(mat0, 1, 1, 4.0);
	CHECK_ERROR();
	MatrixOps::mat_print(mat0);

	printf("\nmat_get()\n");
	VERIFY_MAT_GET(mat0, 0, 0, 1.0);
	VERIFY_MAT_GET(mat0, 0, 1, 2.0);
	VERIFY_MAT_GET(mat0, 1, 0, 3.0);
	VERIFY_MAT_GET(mat0, 1, 1, 4.0);
	printf("Passed.\n");

	printf("\nmult()\n");
	MatrixOps::matrix mat2 = MatrixOps::mult(mat0, mat1);
	CHECK_ERROR();
	MatrixOps::mat_print(mat2);

	printf("\nmult_in_place(2.0)\n");
	MatrixOps::mult_in_place(2.0, mat1);
	CHECK_ERROR();
	MatrixOps::mat_print(mat1);

	printf("\nadd()\n");
	MatrixOps::mat_print(mat0);
	printf("+\n");
	MatrixOps::add(mat0, mat1, mat0, 1.0);
	CHECK_ERROR();
	MatrixOps::mat_print(mat1);
	printf("=\n");
	MatrixOps::mat_print(mat0);

	printf("\nadd()\n");
	MatrixOps::mat_print(mat0);
	printf("-\n");
	MatrixOps::add(mat0, mat1, mat0, -1.0);
	CHECK_ERROR();
	MatrixOps::mat_print(mat1);
	printf("=\n");
	MatrixOps::mat_print(mat0);

	printf("\n\nTesting Matrix-Vector Operations.\n");

	printf("\nmult_vec()\n");
	MatrixOps::vector vec2 = MatrixOps::vec_zeros(2);
	CHECK_ERROR();
	MatrixOps::vec_set(vec2, 0, 4);
	CHECK_ERROR();
	MatrixOps::vec_set(vec2, 1, 8);
	CHECK_ERROR();
	MatrixOps::vector vec3 = MatrixOps::mult_vec(vec2, mat1);
	CHECK_ERROR();
	MatrixOps::vec_print(vec2);
	printf("*\n");
	MatrixOps::mat_print(mat1);
	printf("=\n");
	MatrixOps::vec_print(vec3);

	printf("\nfree_mat() & free_vec()\n");
	MatrixOps::free_mat(mat0);
	CHECK_ERROR();
	MatrixOps::free_mat(mat1);
	CHECK_ERROR();
	MatrixOps::free_mat(mat2);
	CHECK_ERROR();
	MatrixOps::free_vec(vec2);
	CHECK_ERROR();
	MatrixOps::free_vec(vec3);
	CHECK_ERROR();
	printf("Passed\n");

	return true;
}