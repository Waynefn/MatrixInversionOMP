# MatrixInversionOMP

GPU based positive-definite matrix inversion using cholesky decomposition & triangular matrix inversion

author: Zhixin Li

date: 2015/5/19

## Namings in cu

dev\*: device pointer (GPU)

host\*: host pointer (CPU)

n: size of matrix

m: size of submatrix in this algorithm

ld: Storage location row offset (unit is number, here is usually the initial matrix size n)

offset\*: current memory address (offset int)

\*(FuncName)withCuda: internal function that calls kernel

\_(FuncName)withCubl: internal function that calls cublas

\_\_k(C|T)(Name): kelnel function C:Cholesky, T: Triangular inversion
