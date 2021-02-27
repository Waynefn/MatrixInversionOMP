/*************************************
* positive-definite matrix inversion
*
* cholesky decomposition & triangular matrix inversion
*
* Zhixin Li  From 2015/5 /6
*            To   2015/5 /19
**************************************/

/*

Namings:

dev_		device pointer (GPU)
host_		host pointer (CPU)
n			size of matrix
m			size of submatrix in this algorithm
ld			Storage location row offset (unit is number, here is usually the initial matrix size n)
offset_		current memory address (offset int)
_(FuncName)withCuda    internal function that calls kernel
_(FuncName)withCubl    internal function that calls cublas
__k(C|T)(Name)		kelnel function  C:Cholesky, T: Triangular inversion

*/

#include <stdio.h>
// CUDA runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//cuda lib, matrix lib,cublas lib (matrix multiplication)
#include <cublas_v2.h>
#include "matrixb.h"

/ / Define the matrix multiplication thread block size(the block size in the inversion, less than calling less, greater than calling BATA, or changing to cublas matrix array multiplication, this macro has been abandoned)
// Now as the block size of the clear function
// !! Now do the maximum number of concurrent threads for the device dim3 (TILED_SIZE, TILED_SIZE)
#define TILED_SIZE 32
// ! It is very important to define the block size calculated by chol at one time, the total block size, and determine the recursive block size of chol and tmi
#define PDMI_SIZE 256
#define SIZEOF_MATYPE sizeof(matype)
#define SIZEOF_MATYPE_P sizeof(matype *)

	//Matrix calculation constant
	const matype alpha_minus1 = -1.0f;
const matype alpha_plus1 = 1.0f;
const matype beta0 = 0.0f;
const matype beta1 = 1.0f;

__global__ void __kCDiagSqrt(matype *dev_mx, int m, int ld)
{
	dev_mx[m * ld + m] = sqrt(dev_mx[m * ld + m]);
}

__global__ void __kCMatSubdc(matype *dev_mx, int n, int m, int ld)
{
	int i = threadIdx.x + m + 1;
	int j = threadIdx.y + m + 1;
	__shared__ matype k;
	if (j < n)
	{
		if (i == m + 1)
		{
			if (j == m + 1)
			{
				k = sqrt(dev_mx[m * ld + m]);
				dev_mx[m * ld + m] = k;
			}
			__syncthreads();
			dev_mx[m * ld + j] /= k;
		}
		__syncthreads();
		if (j >= i)
			dev_mx[i * ld + j] -= dev_mx[m * ld + i] * dev_mx[m * ld + j];
	}
}

// Find the inverse of the diagonal elements
__global__ void __kTDiagInv(matype *dev_mx, int ld, int n)
{
	const int tid = threadIdx.x;
	if (tid < n)
		dev_mx[tid * ld + tid] = 1 / dev_mx[tid * ld + tid];
}

__global__ void __kTSetLowZero(matype *dev_mx, int n, int ld)
{
	int x = threadIdx.x + blockIdx.x * TILED_SIZE;
	int y = threadIdx.y + blockIdx.y * TILED_SIZE;
	if (x > y && x < n)
		dev_mx[x * ld + y] = 0;
}

int _bTMIwithCubl(matype *dev_mx, matype *dev_resl, int n, int ld, cublasHandle_t handle)
{
	int *host_diad = new int[n];
	int offset_B;
	int offset_A;
	int offset_C;
	int offset_T;

	for (int i = 0; i < n; i++)
		host_diad[i] = ld * i + i;

	dim3 grid_size(n / TILED_SIZE, n / TILED_SIZE);
	dim3 block_size(TILED_SIZE, TILED_SIZE);

	__kTSetLowZero<<<grid_size, block_size>>>(dev_mx, n, ld);
	__kTDiagInv<<<1, n>>>(dev_mx, ld, n);

	for (int m = 1; m < n; m *= 2)
	{
		int BatchedCount = n / (2 * m);

		matype **host_batch_B = new matype *[BatchedCount];
		matype **host_batch_A = new matype *[BatchedCount];
		matype **host_batch_C = new matype *[BatchedCount];
		matype **host_batch_T = new matype *[BatchedCount];

		matype **dev_batch_B = (matype **)&dev_resl[0];
		matype **dev_batch_A = (matype **)&dev_resl[n];
		matype **dev_batch_C = (matype **)&dev_resl[2 * n];
		matype **dev_batch_T = (matype **)&dev_resl[3 * n];

		for (int i = 0; i < BatchedCount; i++)
		{
			offset_B = host_diad[2 * i * m];
			offset_A = offset_B + m;
			offset_C = host_diad[2 * i * m + m];
			offset_T = offset_C - m;

			host_batch_B[i] = &dev_mx[offset_B];
			host_batch_A[i] = &dev_mx[offset_A];
			host_batch_C[i] = &dev_mx[offset_C];
			host_batch_T[i] = &dev_mx[offset_T];
		}

		cudaMemcpy(
			dev_batch_B,
			host_batch_B,
			SIZEOF_MATYPE_P * BatchedCount,
			cudaMemcpyHostToDevice);

		cudaMemcpy(
			dev_batch_A,
			host_batch_A,
			SIZEOF_MATYPE_P * BatchedCount,
			cudaMemcpyHostToDevice);

		cudaMemcpy(
			dev_batch_T,
			host_batch_T,
			SIZEOF_MATYPE_P * BatchedCount,
			cudaMemcpyHostToDevice);

		cudaMemcpy(
			dev_batch_C,
			host_batch_C,
			SIZEOF_MATYPE_P * BatchedCount,
			cudaMemcpyHostToDevice);

		cublasSgemmBatched(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			m, m, m,
			&alpha_minus1,
			(const matype **)dev_batch_A, ld,
			(const matype **)dev_batch_B, ld,
			&beta0,
			dev_batch_T, ld,
			BatchedCount);

		cublasSgemmBatched(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			m, m, m,
			&alpha_plus1,
			(const matype **)dev_batch_C, ld,
			(const matype **)dev_batch_T, ld,
			&beta0,
			dev_batch_A, ld,
			BatchedCount);

		__kTSetLowZero<<<grid_size, block_size>>>(dev_mx, n, ld);

		delete host_batch_A;
		delete host_batch_B;
		delete host_batch_C;
		delete host_batch_T;
	}

	delete[] host_diad;
	return SUCCESS;
}

int potrf(matype *dev_mx, int n, int ld)
{
	dim3 blocks(n, n);

	for (int i = 0; i < n - 1; i++)
		__kCMatSubdc<<<1, blocks>>>(dev_mx, n, i, ld);
	__kCDiagSqrt<<<1, 1>>>(dev_mx, n - 1, ld);

	return SUCCESS;
}

int _bCHOLwithCubl(matype *dev_mx, int n, int ld, cublasHandle_t handle)
{
	int BlockWidth = n / TILED_SIZE;
	int *offset = new int[BlockWidth];

	for (int i = 0; i < BlockWidth; i++)
		offset[i] = i * TILED_SIZE * ld + i * TILED_SIZE;

	for (int i = 0; i < BlockWidth - 1; i++)
	{
		potrf(
			&dev_mx[offset[i]],
			TILED_SIZE,
			ld);

		cublasStrsm(
			handle,
			CUBLAS_SIDE_RIGHT,
			cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
			cublasOperation_t::CUBLAS_OP_T,
			cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
			(BlockWidth - i - 1) * TILED_SIZE,
			TILED_SIZE,
			&alpha_plus1,
			&dev_mx[offset[i]], ld,
			&dev_mx[offset[i] + TILED_SIZE], ld);

		cublasSsyrk(
			handle,
			cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N,
			(BlockWidth - i - 1) * TILED_SIZE,
			TILED_SIZE,
			&alpha_minus1,
			&dev_mx[offset[i] + TILED_SIZE], ld,
			&beta1,
			&dev_mx[offset[i + 1]], ld);
	}
	potrf(
		&dev_mx[offset[BlockWidth - 1]],
		TILED_SIZE,
		ld);

	delete[] offset;
	return SUCCESS;
}

/*
S=B(-T)A  C~=C^-A(T)A
| B  A |
| T  C |
*/
int _MMulBTAAwithCuda(matype *dev_b, matype *dev_a, matype *dev_c, matype *dev_t, int n, int ld, cublasHandle_t handle)
{
	dim3 grid_size(n / TILED_SIZE, n / TILED_SIZE);
	dim3 block_size(TILED_SIZE, TILED_SIZE);
	__kTSetLowZero<<<grid_size, block_size>>>(dev_b, n, ld);

	cublasSgemm(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_T,
		n, n, n,
		&alpha_plus1,
		dev_t, ld,
		dev_b, ld,
		&beta0,
		dev_a, ld);

	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		n, n, n,
		&alpha_minus1,
		dev_a, ld,
		dev_a, ld,
		&beta1,
		dev_c, ld);
	return SUCCESS;
}

int _FinMulwithCuda(matype *dev_mx, matype *dev_resl, int n, int ld, cublasHandle_t handle)
{
	dim3 grid_size(n / TILED_SIZE, n / TILED_SIZE);
	dim3 block_size(TILED_SIZE, TILED_SIZE);
	__kTSetLowZero<<<grid_size, block_size>>>(dev_mx, n, n);

	cublasStrmm_v2(handle,
				   CUBLAS_SIDE_LEFT,
				   cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
				   cublasOperation_t::CUBLAS_OP_T,
				   cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
				   n, n,
				   &alpha_plus1,
				   dev_mx, n,
				   dev_mx, n,
				   dev_resl, n);
	//int a=cublasSsyrk_v2(
	//handle,
	//cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
	//cublasOperation_t::CUBLAS_OP_T,
	//n, n,
	//&alpha_plus1,
	//dev_mx,	ld,
	//&beta0,
	//dev_resl,ld);
	return SUCCESS;
}

/*
	A=-B*A*C
	| B  A |
	| T  C |

*/
int _MMulBACwithCuda(matype *dev_b, matype *dev_a, matype *dev_c, matype *dev_t, int n, int ld, cublasHandle_t handle)
{

	dim3 grid_size(n / TILED_SIZE, n / TILED_SIZE);
	dim3 block_size(TILED_SIZE, TILED_SIZE);
	__kTSetLowZero<<<grid_size, block_size>>>(dev_c, n, ld);

	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		n, n, n,
		&alpha_minus1,
		dev_a, ld,
		dev_b, ld,
		&beta0,
		dev_t, ld);

	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		n, n, n,
		&alpha_plus1,
		dev_c, ld,
		dev_t, ld,
		&beta0,
		dev_a, ld);

	return SUCCESS;
}

int _cudaInitial(int devID = 0)
{
	cudaDeviceProp deviceProp;
	int error = cudaGetDeviceProperties(&deviceProp, devID);
	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	return error;
}

/*
| B  A |
| T  C |
*/
void _bRecursive(matype *dev_mx, matype *dev_resl, int n_now, int ld, cublasHandle_t handle)
{
	if (n_now == PDMI_SIZE)
	{
		_bCHOLwithCubl(dev_mx, n_now, ld, handle);
		_bTMIwithCubl(dev_mx, dev_resl, n_now, ld, handle);
	}
	else
	{
		matype *dev_b = dev_mx;
		matype *dev_a = &dev_mx[n_now / 2];
		matype *dev_t = &dev_mx[n_now * ld / 2];
		matype *dev_c = &dev_t[n_now / 2];

		_bRecursive(dev_b, dev_resl, n_now / 2, ld, handle);
		_MMulBTAAwithCuda(dev_b, dev_a, dev_c, dev_t, n_now / 2, ld, handle);
		_bRecursive(dev_c, dev_resl, n_now / 2, ld, handle);
		_MMulBACwithCuda(dev_b, dev_a, dev_c, dev_t, n_now / 2, ld, handle);
	}
	return;
}

int SetZero(matype *dev_mx, char *mode, int n, int ld)
{
	switch (*mode)
	{
	case 'L':

		__kTSetLowZero<<<dim3(n / TILED_SIZE, n / TILED_SIZE), dim3(TILED_SIZE, TILED_SIZE)>>>(dev_mx, n, ld);
		break;
	case 'U':
		// dim3 grid_size2(n / TILED_SIZE, n / TILED_SIZE);
		// dim3 block_size2(TILED_SIZE, TILED_SIZE);
		//__kTSetLowZero << <grid_size, block_size >> >(dev_mx, n, ld);
		break;

	case 'A':

		break;
	default:
		return FAIL;
		break;
	}
	return SUCCESS;
}

int WriteLog(char *path, int n, float time_use)
{
	FILE *fp;
	if ((fp = fopen(path, "at")) == NULL)
		return FAIL;
	fprintf(fp, "VS,%d,%.3f,VS\n", n, time_use);
	fclose(fp);
	return SUCCESS;
}

int main(int argc, char *argv[])
{
	matype *host_mx;
	int n;
	// Define the amount of test data

	int $i;
	printf("Input COUNT:");
	scanf("%d", &$i);
	// Process the main function to pass parameters

	if (argc == 2)
	{
		n = atoi(argv[1]);
		if (n < 1 || n > 8192)
			n = 256;
		printf("Argc Matrix order N:%d\n", n);
	}
	else
	{
		printf("Input order N:");
		scanf("%d", &n);
	}
	CreatMat(host_mx, n);
	puts("Positive Definite Matrix Inversion using GPU");

	////////////////////////////////////////////
	clock_t cl_st = clock();
	matype *dev_mx;
	matype *dev_resl;
	cublasHandle_t handle;
	cudaEvent_t start, stop;
	float msecTotal = 0.0f;

	// Initialize variables

	_cudaInitial(0);
	cublasCreate(&handle);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory to 2^k n
	cudaMalloc(
		(void **)&dev_mx,
		n * n * SIZEOF_MATYPE);

	cudaMalloc(
		(void **)&dev_resl,
		n * n * SIZEOF_MATYPE);

	//cudaMemset(
	//	dev_mx,
	//	0,
	//	n*n*SIZEOF_MATYPE);

	char path[100];
	for (int count = 0; count <= $i; count++)
	{
		sprintf(path, "PM[%d][%d].dat", n, count);
		ReadMatf(path, host_mx, n);

		cudaMemcpy2D(
			dev_mx,
			SIZEOF_MATYPE * n,
			host_mx,
			SIZEOF_MATYPE * n,
			SIZEOF_MATYPE * n,
			n,
			cudaMemcpyHostToDevice);

		printf("Calculate with GPU...");
		cudaEventRecord(start, NULL);

		// Call kernel function

		_bRecursive(dev_mx, dev_resl, n, n, handle);
		_FinMulwithCuda(dev_mx, dev_resl, n, n, handle);

		cudaEventRecord(stop, NULL);
		// Wait for the stop event to complete
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&msecTotal, start, stop);
		// End processing

		printf("done\nGPU Time=%.3fmsec\n", msecTotal);

		WriteLog("cudaINV_Omega.log", n, msecTotal);
		cudaMemcpy2D(
			host_mx,
			SIZEOF_MATYPE * n,
			dev_resl,
			n * SIZEOF_MATYPE,
			n * SIZEOF_MATYPE,
			n,
			cudaMemcpyDeviceToHost);
		sprintf(path, "D:\\developer\\cudaINV_Omega\\cudaINV_Omega\\VP[%d][%d].dat", n, count);
		WriteMatf(host_mx, path, n);
	}
	cudaFree(dev_mx);
	cudaFree(dev_resl);
	cublasDestroy(handle);

	cudaDeviceReset();
	clock_t cl_et = clock();
	////////////////////////////////////////////

	FreeMat(host_mx, n);
	printf("Cpu Time=%.3fmsec\n", (float)(cl_et - cl_st) / ($i + 1));
	puts("Quit Success");
	return 0;
}
