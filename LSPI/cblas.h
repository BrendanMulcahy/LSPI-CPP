/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t  /* this may vary between platforms */

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#ifdef __cplusplus
extern "C" {
#endif

void sgemm_(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
            const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, 
			const int ldc);

void sgemv_(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const float *A, 
			const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);

void sscal_(const int N, const float alpha, float *X, const int incX);

float sdot_(const int N, const float  *X, const int incX, const float  *Y, const int incY);

void saxpy_(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);

#ifdef __cplusplus
}
#endif