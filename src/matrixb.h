// Matrix custom library function
// Matrix is ??mapped to one dimension
// Zhixin Li
// 2015 04 28 update

#pragma once

#ifndef MATRIXB_H
#define MATRIXB_H
#include <math.h>
#include <stdlib.h>
#include <time.h>
extern "C"
{

// success code
#define SUCCESS 0
// fail code
#define FAIL 1
// output format
#define PRINT_FORMAT "%.5g\t"
// random seed
#define RND_SEED (unsigned)time(NULL)
// rand max: 0~RND_MAX
#define RND_MAX 1

	// precision
	typedef float matype;

#define RND_PARA (matype) RAND_MAX *RND_MAX
	// output matrix
	int PrintMat(matype *Matrix, int n, char *prompt);

	// init matrix
	int CreatMat(matype *&Matrix, int n);

	// assign unit matrix
	int EyeAssignMat(matype *Matrix, int n);

	// assign random matrix
	int RndAssignMat(matype *mx, int n);

	// free matrix pointer
	int FreeMat(matype *&mx, int n);

	// random assign symmetric matrix
	int RndAssignMatSymmtr(matype *mx, int n);

	// randomly assign triangular matrix
	int RndAssignMatTriangl(matype *mx, int n, bool IsUpper);

	// assign positive-definite matrix
	int RndAssignMatPosDefnt(matype *mx, int n);

	// Use elementary transformation to find the inverse matrix
	int inv(matype *mx, int n);

	// Use elementary transformation to find the inverse matrix (2)
	int inv2(matype *mx, matype *mx_inv, int n);

	// Cholesky decomposition
	int chol(matype *mx, int n);

	// Cholesky decomposition (2)
	int chol2(matype *mx, int n);

	// copy matrix
	int CopyMat(matype *dst, matype *src, int n);

	// error analysis. a, b are both lower triangular matrix
	int CmpMatLL(const matype *a, const matype *b, int n);

	// error analysis. a is the lower triangular matrix, b is the upper triangular matrix,
	int CmpMatLU(const matype *a, const matype *b, int n);

	// write matrix to file
	int WriteMatf(matype *mx, char *path, int n);

	// load file to memory
	int ReadMatf(char *path, matype *mx, int n);
}

#endif
