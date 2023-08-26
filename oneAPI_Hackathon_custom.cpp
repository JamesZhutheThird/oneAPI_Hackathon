/*
使用oneMKL工具，对FFT算法进行加速与优化。
1. 点击链接下载和使用最新版本oneMKL
2. 调用 oneMKL 相应 API函数， 产生 2048 * 2048 个 随机单精度实数()；
3. 根据2产生的随机数据作为输入，实现两维 Real to complex FFT 参考代码；
4. 根据2产生的随机数据作为输入， 调用 oneMKL API 计算两维 Real to complex FFT；
5. 结果正确性验证，对3和4计算的两维FFT输出数据进行全数据比对（允许适当精度误差）， 输出 “结果正确”或“结果不正确”信息；
6. 平均性能数据比对（比如运行1000次），输出FFT参考代码平均运行时间和 oneMKL FFT 平均运行时间。
*/

// 添加头文件
#include <stdio.h>
#include <mkl.h>
#include <chrono>
#include <iostream>
#include <complex>
#include <cmath>
#include <fftw3.h>
#include <vector>
using namespace std;

// 定义宏
#define M 8
#define N 8
#define PRINT_LIMIT 10
#define PI acos(-1.0)
#define EPSILON 1e-6
#define REPEAT 1000
#define RUN_REPEAT false

typedef complex<float> Complex;

// 输出矩阵
void print_matrix(float* matrix_to_print, int m, int n, int limit = 0)
{
	int print_m = m;
	int print_n = n;
	if (limit > 0)
	{
		print_m = min(m, limit);
		print_n = min(n, limit);
	}

	for (int i = 0; i < print_m; i++)
	{
		for (int j = 0; j < print_n; j++)
			printf("%.3f  ", matrix_to_print[i * n + j]);
		printf((print_m == m) ? "\n" : "...\n");
	}
	printf((print_n == n) ? "\n" : "...\n\n");
}

void print_matrix_complex(Complex* matrix_to_print, int m, int n, int limit = 0)
{
	int print_m = m;
	int print_n = n;
	if (limit > 0)
	{
		print_m = min(m, limit);
		print_n = min(n, limit);
	}

	for (int i = 0; i < print_m; i++)
	{
		for (int j = 0; j < print_n; j++)
			printf("%.3f + %.3fi  ", matrix_to_print[i * n + j].real(), matrix_to_print[i * n + j].imag());
		printf((print_m == m) ? "\n" : "...\n");
	}
	printf((print_n == n) ? "\n" : "...\n\n");
}


// 随机生成单精度实数
float matrix_[M * N]={0.639, 0.111, 0.025, 0.742, 0.275, 0.245, 0.223, 0.140,
	0.736, 0.102, 0.677, 0.741, 0.892, 0.545, 0.087, 0.590,
	0.422, 0.032, 0.030, 0.094, 0.219, 0.233, 0.505, 0.602,
	0.027, 0.561, 0.199, 0.716, 0.650, 0.701, 0.545, 0.420,
	0.220, 0.449, 0.589, 0.278, 0.809, 0.869, 0.006, 0.759,
	0.806, 0.160, 0.698, 0.423, 0.340, 0.278, 0.155, 0.215,
	0.957, 0.763, 0.337, 0.102, 0.093, 0.380, 0.097, 0.359,
	0.847, 0.344, 0.604, 0.265, 0.807, 0.043, 0.730, 0.459};
float* matrix = matrix_;

// 2D Real to complex FFT using recursive method
Complex* matrix_fft_custom = (Complex*)malloc(sizeof(Complex) * M * N);/* save matrix after fft_custom */


// Calculate FFT using recursive method
void fft1d_custom(Complex* mat, int size, int stride) {
	if (size <= 1)
		return;

	// Divide the array into even and odd parts
	int halfSize = size / 2;
	Complex* even = (Complex*)malloc(sizeof(Complex) * halfSize);
	Complex* odd = (Complex*)malloc(sizeof(Complex) * halfSize);

	for (int i = 0; i < halfSize; ++i) 
	{
		// cout << mat[i * stride] << '\t' << mat[(i * stride) + stride] <<'\n' << endl;
		even[i] = mat[i * stride];
		odd[i] = mat[(i * stride) + stride];
	}

	// Recursively calculate FFT for even and odd parts

	fft1d_custom(even, halfSize, stride * 2);
	fft1d_custom(odd, halfSize, stride * 2);

	// Combine the results
	for (int i = 0; i < size / 2; ++i) 
	{
		Complex t= { 0, 0 };
		float angle = -2 * PI * i / size;

		t.real(cos(angle) * odd[i].real() + sin(angle) * odd[i].imag());
		t.imag(cos(angle) * odd[i].imag() - sin(angle) * odd[i].real());
		
		mat[i * stride].real(even[i].real() + t.real());
		mat[i * stride].imag(even[i].imag() + t.imag());

		mat[(i + size / 2) * stride].real(even[i].real() - t.real());
		mat[(i + size / 2) * stride].imag(even[i].imag() - t.imag());
	}
}

// Perform 2D FFT using recursive method
void fft2d_custom(Complex* mat) {
	// Apply FFT on rows
	for (int i = 0; i < M; ++i)
		fft1d_custom(mat + i * N, N, 1);

	// Apply FFT on columns
	for (int i = 0; i < N; ++i) 
	{
		Complex* column = (Complex*)malloc(sizeof(Complex) * M);
		for (int j = 0; j < M; ++j)
			column[j] = mat[j * N + i];
		fft1d_custom(column, M, 1);
		for (int j = 0; j < M; ++j) 
			mat[j * N + i] = column[j];
	}
}

// wrapper for fft2d_custom
void fft_custom()
{
	// copy data
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			matrix_fft_custom[i * N + j] = matrix[i * N + j];

	// fft
	fft2d_custom(matrix_fft_custom);
}

// 2D Real to complex FFT with fftw3.h
float* matrix_fft_fftw3 = (float*)fftwf_malloc(sizeof(float) * 2 * M * (N / 2 + 1));/* save matrix after fft_fftw3 */
Complex* matrix_fft_fftw3_trans = (Complex*)malloc(sizeof(Complex) * M * N); /* decode matrix after fft_fftw3 */

void fft_fftw3()
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_r2c_2d(M, N, matrix, (fftwf_complex*)matrix_fft_fftw3, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

void fft_fftw3_transform()
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			if (i == 0)
				if (j < (N / 2 + 1))
				{
					matrix_fft_fftw3_trans[i * N + j].real(matrix_fft_fftw3[j * 2]);
					matrix_fft_fftw3_trans[i * N + j].imag(matrix_fft_fftw3[j * 2 + 1]);
				}
				else
				{
					matrix_fft_fftw3_trans[i * N + j].real(matrix_fft_fftw3[(N - j) * 2]);
					matrix_fft_fftw3_trans[i * N + j].imag(-matrix_fft_fftw3[(N - j) * 2 + 1]);
				}
			else
				if (j < (N / 2 + 1))
				{
					matrix_fft_fftw3_trans[i * N + j].real(matrix_fft_fftw3[(i * (N / 2 + 1) + j) * 2]);
					matrix_fft_fftw3_trans[i * N + j].imag(matrix_fft_fftw3[(i * (N / 2 + 1) + j) * 2 + 1]);
				}
				else
				{
					matrix_fft_fftw3_trans[i * N + j].real(matrix_fft_fftw3[((M - i) * (N / 2 + 1) + (N - j)) * 2]);
					matrix_fft_fftw3_trans[i * N + j].imag(-matrix_fft_fftw3[((M - i) * (N / 2 + 1) + (N - j)) * 2 + 1]);
				}
}


// 验证结果
float verify(Complex* mat_1, Complex* mat_2, float epsilon = 1e-5)
{
	float max_diff = 0;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			max_diff = max(max_diff, abs(mat_1[i * N + j].real() - mat_2[i * N + j].real()));
			max_diff = max(max_diff, abs(mat_1[i * N + j].imag() - mat_2[i * N + j].imag()));
		}

	return max_diff;
}



// 主程序
int main()
{
	// 生成随机数
	printf("生成随机数\n");
	printf("随机数为：\n");
	print_matrix(matrix, M, N, PRINT_LIMIT);

	// custom
	printf("使用自定义函数进行FFT\n");
	fft_custom();
	print_matrix_complex(matrix_fft_custom, M, N, PRINT_LIMIT);

	// fftw3
	printf("使用 fftw3 进行FFT\n");
	fft_fftw3();
	print_matrix(matrix_fft_fftw3, M, 2 * (N / 2 + 1), PRINT_LIMIT);
	fft_fftw3_transform();
	print_matrix_complex(matrix_fft_fftw3_trans, M, N, PRINT_LIMIT);

	// 验证结果
	printf("最大误差为%f\n", verify(matrix_fft_fftw3_trans, matrix_fft_custom));
	if (verify(matrix_fft_fftw3_trans, matrix_fft_custom) < EPSILON)
		printf("结果正确\n");
	else
		printf("结果不正确\n");

	return 0;
}