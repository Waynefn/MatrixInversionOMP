// Matrix library functions, see the header file for details
#include <stdio.h>
// #include"stdafx.h"
#include "matrixb.h"

int PrintMat(matype *mx, int n, char *prompt)
{
	printf("%s matrix[%d]:\n", prompt, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			printf(PRINT_FORMAT, mx[i * n + j]);
		putchar('\n');
	}
	return SUCCESS;
}

int CreatMat(matype *&mx, int n)
{
	srand(RND_SEED);
	mx = new matype[n * n];
	return SUCCESS;
}

int FreeMat(matype *&mx, int n)
{
	delete[] mx;
	return SUCCESS;
}

int EyeAssignMat(matype *mx, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			if (i == j)
				mx[i * n + j] = 1;
			else
				mx[i * n + j] = 0;
		}
	return SUCCESS;
}

int RndAssignMat(matype *mx, int n)
{

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			mx[i * n + j] = (matype)rand() / RND_PARA + 10;
		}
	return SUCCESS;
}

int RndAssignMatSymmtr(matype *mx, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j <= i; j++)
		{
			matype r = (matype)rand() / RND_PARA;
			mx[i * n + j] = r;
			mx[j * n + i] = r;
		}
	return SUCCESS;
}

int RndAssignMatTriangl(matype *mx, int n, bool IsUpper)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j <= i; j++)
		{
			matype r = (matype)rand() / RND_PARA;
			if (!IsUpper)
			{
				mx[j * n + i] = 0;
				mx[i * n + j] = r;
			}
			else
			{
				mx[i * n + j] = 0;
				mx[j * n + i] = r;
			}
		}
	return SUCCESS;
}

int RndAssignMatPosDefnt(matype *mx, int n)
{
	matype *rndmx;
	CreatMat(rndmx, n);
	RndAssignMat(rndmx, n);
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
		{
			matype _sum = 0.0;
			for (int k = 0; k < n; k++)
				_sum += rndmx[i * n + k] * rndmx[j * n + k];
			mx[i * n + j] = _sum;
			mx[j * n + i] = _sum;
		}
	FreeMat(rndmx, n);
	return SUCCESS;
}

int inv2(matype *mx, matype *mx_inv, int n)
{
	matype k;
	for (int i = 0; i < n; i++)
	{
		k = 1 / mx[i * n + i];
		for (int j = 0; j < n; j++)
		{
			mx[i * n + j] *= k;
			mx_inv[i * n + j] *= k;
		}
		for (int j = 0; j < n; j++)
		{
			if (j == i)
				continue;
			else
				k = -mx[j * n + i];
			for (int w = 0; w < n; w++)
			{
				mx[j * n + w] += k * mx[i * n + w];
				mx_inv[j * n + w] += k * mx_inv[i * n + w];
			}
		}
	}
	return SUCCESS;
}

int inv(matype *mx, int n)
{
	for (int i = 0; i < n; i++)
	{
		double k1 = 1 / mx[i * n + i];
		mx[i * n + i] = 1.0;

		for (int l = 0; l < n; l++)
			mx[i * n + l] *= k1;
		for (int j = 0; j < n; j++)
		{
			double k2 = mx[j * n + i];
			if (j != i)
				for (int l = 0; l < n; l++)
				{
					if (l == i)
						mx[j * n + l] *= -k1;
					else
						mx[j * n + l] -= k2 * mx[i * n + l];
				}
		}
	}
	return SUCCESS;
}

int chol(matype *mx, int n)
{
	for (int k = 0; k < n; k++)
	{
		if (mx[k * n + k] < 0)
			return FAIL;
		matype kk = sqrt(mx[k * n + k]);
		for (int i = k; i < n; i++)
			mx[i * n + k] /= kk;

		for (int j = k + 1; j < n; j++)
		{
			matype jk = mx[j * n + k];
			for (int i = j; i < n; i++)
				mx[i * n + j] -= mx[i * n + k] * jk;
		}
	}

	return SUCCESS;
}

int chol2(matype *mx, int n)
{
	matype Rkk;
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			matype Rkj = mx[k * n + j];
			Rkk = mx[k * n + k];
			for (int h = j; h < n; h++)
				mx[j * n + h] -= mx[k * n + h] * Rkj / Rkk;
		}
		Rkk = sqrt(mx[k * n + k]);
		for (int h = k; h < n; h++)
			mx[k * n + h] /= Rkk;
	}
	return SUCCESS;
}

int CopyMat(matype *dst, matype *src, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			dst[i * n + j] = src[i * n + j];
		}
	return SUCCESS;
}

int CmpMatLU(const matype *a, const matype *b, int n)
{
	float max_err = 0.0;
	float average_err = 0.0;
	int i, j;
	for (i = 0; i < n; i++)
		for (j = i; j < n; j++)
		{
			if (b[i * n + j] != 0.0)
			{
				float err = fabsf((a[j * n + i] - b[i * n + j]) / b[i * n + j]);
				//assert(!isnan(err));
				//if (isnan(err))
				//{
				//	k++;
				//}
				if (max_err < err)
				{
					max_err = err;
				}
				average_err += err;
			}
		}
	printf("max error=%e average error=%e\n", max_err, average_err / (n * n));
	return SUCCESS;
}

int CmpMatLL(const matype *a, const matype *b, int n)
{
	float max_err = 0.0;
	float average_err = 0.0;
	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j <= i; j++)
		{
			if (b[i * n + j] != 0.0)
			{
				float err = fabs((a[i * n + j] - b[i * n + j]) / b[i * n + j]);
				//assert(!isnan(err));
				//if (isnan(err))
				//{
				//	k++;
				//}
				if (max_err < err)
					max_err = err;
				average_err += err;
			}
		}
	printf("Max error=%e Average error=%e\n", max_err, average_err / (n * n));
	return SUCCESS;
}

int WriteMatf(matype *mx, char *path, int n)
{
	FILE *fp;
	if ((fp = fopen(path, "wb")) == NULL)
		return FAIL;
	fwrite(mx, sizeof(matype), n * n, fp);
	fclose(fp);
	return SUCCESS;
}

int ReadMatf(char *path, matype *mx, int n)
{
	FILE *fp;
	if ((fp = fopen(path, "rb")) == NULL)
		return FAIL;
	fread(mx, sizeof(matype), n * n, fp);
	fclose(fp);
	return SUCCESS;
}
