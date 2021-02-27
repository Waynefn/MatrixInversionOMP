// matrix_line.cpp 
//

#include "stdafx.h"

#include<time.h>
#include<omp.h>
#define MTX_ODR 500
#define TIMES 10
#define THREAD_NUM 4

void invKernel(double*dData, int nOrder)
{
	//omp_set_num_threads(THREAD_NUM);
	double k1, k2;
	int i, j, k;
//#pragma omp parallel
//	{
//#pragma omp for
		for (int i = 0; i < nOrder; i++)
		{
			int i_ROW = i*nOrder;
			k1 = 1.0 / dData[i_ROW + i];
			dData[i_ROW + i] = 1.0;

			for (int k = 0; k < nOrder; k++)
				dData[i_ROW + k] *= k1;

			for (int j = 0; j < nOrder; j++)
			{
				if (j == i)continue;
				int j_ROW = j*nOrder;
				k2 = dData[j_ROW + i];
				dData[j_ROW + i] *= -k1;

				for (int k = 0; k < i; k++)
					dData[j_ROW + k] -= k2* dData[i_ROW + k];

				for (int k = i + 1; k < nOrder; k++)
					dData[j_ROW + k] -= k2* dData[i_ROW + k];
			}
		//}
	}
}
int PrintMatrix(MATRIX_L*m)
{
	printf("matrix[%d*%d]:\n", m->nOrder, m->nOrder);
	for (int i = 0; i < m->nOrder; i++)
	{
		for (int j = 0; j < m->nOrder; j++)
			printf(PRINT_FORMAT, m->dData[i*m->nOrder + j]);
		putchar('\n');
	}
	return 1;
}
int InitialMatrix(int odr, MATRIX_L*&d)
{
	d = new MATRIX_L;
	d->nOrder = odr;
	d->dData = new double[odr*odr];
	return 1;
}
int DestoryMatrix(MATRIX_L*&d)
{
	delete d->dData;
	delete d;
	return 1;
}
int RandomAssign(MATRIX_L*m)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < m->nOrder; i++)
	for (int j = 0; j < m->nOrder; j++)
	{
		m->dData[i*m->nOrder + j] = rand() / RANDOM_MAX;
	}
	return 1;
}
int RandomAssignSymmetric(MATRIX_L*m)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < m->nOrder; i++)
	for (int j = 0; j <= i; j++)
	{
		double r = rand() % RANDOM_MAX;
		m->dData[i*m->nOrder + j] = r;
		m->dData[j*m->nOrder + i] = r;
	}
	return 1;
}
int main()
{
	MATRIX_L* m;
	InitialMatrix(MTX_ODR, m);
	RandomAssignSymmetric(m);
	//PrintMatrix(m);

	double time_start = clock();
	for (int i = 0; i < TIMES; i++)
	{
		RandomAssignSymmetric(m);
		invKernel(m->dData, m->nOrder);
		
	}
	double time_end = clock();

	//PrintMatrix(m);
	printf("report: order=%d,time=%.5fms\n", MTX_ODR, (time_end - time_start) / TIMES);
	DestoryMatrix(m);
}
