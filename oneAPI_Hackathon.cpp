/*
英特尔 oneAPI 黑客松
JamesZhuthethird
20230827

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
#include <mkl.h>
#include <fftw3.h>
#include <vector>
#include <fstream>
#include <iomanip>
#pragma warning(disable:4996)
using namespace std;

// 定义宏
#define M 2048
#define N 2048
#define PRINT_LIMIT 10
#define PI acos(-1.0)
#define EPSILON 1e-3
#define REPEAT 1000
#define RUN_REPEAT true

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
float* matrix = (float*)mkl_malloc(sizeof(float) * M * N, 64);/* buffer for random numbers */

void generate_matrix(int seed = 42)
{
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M * N, matrix, 0.0f, 1.0f);
	vslDeleteStream(&stream);
}

// 2D Real to complex FFT using recursive method
Complex* matrix_fft_custom = (Complex*)malloc(sizeof(Complex) * M * N);/* save matrix after fft_custom */


// Calculate FFT using recursive method
void fft1d_custom(Complex* mat, int size)
{
	if (size <= 1)
		return;

	// Divide the array into even and odd parts
	int halfSize = size / 2;
	Complex* even = (Complex*)malloc(sizeof(Complex) * halfSize);
	Complex* odd = (Complex*)malloc(sizeof(Complex) * halfSize);

	for (int i = 0; i < halfSize; ++i)
	{
		even[i] = mat[i * 2];
		odd[i] = mat[(i * 2) + 1];
	}

	// Recursively calculate FFT for even and odd parts

	fft1d_custom(even, halfSize);
	fft1d_custom(odd, halfSize);

	// Combine the results
	for (int i = 0; i < halfSize; ++i)
	{
		Complex t = { 0, 0 };
		float angle =  2 * PI * i / size;

		t.real(cos(angle) * odd[i].real() + sin(angle) * odd[i].imag());
		t.imag(cos(angle) * odd[i].imag() - sin(angle) * odd[i].real());

		mat[i].real(even[i].real() + t.real());
		mat[i].imag(even[i].imag() + t.imag());

		mat[i + halfSize].real(even[i].real() - t.real());
		mat[i + halfSize].imag(even[i].imag() - t.imag());
	}

	free(even);
	free(odd);
}

// Perform 2D FFT using recursive method
void fft2d_custom(Complex* mat)
{
	// Apply FFT on rows
	for (int i = 0; i < M; ++i)
		fft1d_custom(mat + i * N, N);

	// Apply FFT on columns
	for (int i = 0; i < N; ++i)
	{
		Complex* column = (Complex*)malloc(sizeof(Complex) * M);
		for (int j = 0; j < M; ++j)
			column[j] = mat[j * N + i];
		fft1d_custom(column, M);
		for (int j = 0; j < M; ++j)
			mat[j * N + i] = column[j];
		free(column);
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
Complex* matrix_fft_fftw3_trans = (Complex*)fftwf_malloc(sizeof(Complex) * M * N); /* decode matrix after fft_fftw3 */

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


// 2D Real to complex FFT with oneMKL
Complex* matrix_fft_mkl = (Complex*)mkl_malloc(sizeof(Complex) * M * N, 64); /* save matrix after fft_mkl */
Complex* matrix_fft_mkl_trans = (Complex*)mkl_malloc(sizeof(Complex) * M * N, 64); /* decode matrix after fft_mkl */

void fft_mkl()
{
	DFTI_DESCRIPTOR_HANDLE hand;
	MKL_LONG matrix_sizes[2] = { M, N };
	DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 2, matrix_sizes);
	DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	DftiCommitDescriptor(hand);
	DftiComputeForward(hand, matrix, matrix_fft_mkl);
	DftiFreeDescriptor(&hand);
}

void fft_mkl_transform()
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			if (i == 0)
				if (j < (N / 2 + 1))
				{
					matrix_fft_mkl_trans[i * N + j].real(matrix_fft_mkl[j].real());
					matrix_fft_mkl_trans[i * N + j].imag(matrix_fft_mkl[j].imag());
				}
				else
				{
					matrix_fft_mkl_trans[i * N + j].real(matrix_fft_mkl[N - j].real());
					matrix_fft_mkl_trans[i * N + j].imag(-matrix_fft_mkl[N - j].imag());
				}
			else
				if (j < (N / 2 + 1))
				{
					matrix_fft_mkl_trans[i * N + j].real(matrix_fft_mkl[i * N + j].real());
					matrix_fft_mkl_trans[i * N + j].imag(matrix_fft_mkl[i * N + j].imag());
				}
				else
				{
					matrix_fft_mkl_trans[i * N + j].real(matrix_fft_mkl[(M - i) * N + (N - j)].real());
					matrix_fft_mkl_trans[i * N + j].imag(-matrix_fft_mkl[(M - i) * N + (N - j)].imag());
				}
}

// 验证结果
float verify(Complex* mat_1, Complex* mat_2)
{
	float current_diff= 0;
	float diff_real = 0;
	float diff_imag = 0;
	float max_diff_info[7] = { 0 }; 

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			diff_real = abs(mat_1[i * N + j].real() - mat_2[i * N + j].real()) / max({ 1.0f, abs(mat_1[i * N + j].real()), abs(mat_1[i * N + j].real() ) });
			diff_imag = abs(mat_1[i * N + j].imag() - mat_2[i * N + j].imag()) / max({1.0f, abs(mat_1[i * N + j].imag()), abs(mat_1[i * N + j].imag() ) });
			current_diff = max(diff_real, diff_imag);
			if (max_diff_info[0] < current_diff)
			{
				max_diff_info[0] = current_diff;
				max_diff_info[1] = i;
				max_diff_info[2] = j;
				max_diff_info[3] = mat_1[i * N + j].real();
				max_diff_info[4] = mat_1[i * N + j].imag();
				max_diff_info[5] = mat_2[i * N + j].real();
				max_diff_info[6] = mat_2[i * N + j].imag();
			}
			
		}

	if (max_diff_info[0] == 0)
		printf("两矩阵数据完全一致\n"); 
	else 
	{ 
		if (max_diff_info[0] <= EPSILON)
			printf("两矩阵数据差异处于误差范围内\n");
		else
			printf("两矩阵数据差异超出误差范围\n");
		printf("最大差值为%f, 位于(%d, %d), mat_1(%f, %f), mat_2(%f, %f)\n", max_diff_info[0], max_diff_info[1], max_diff_info[2], max_diff_info[3], max_diff_info[4], max_diff_info[5], max_diff_info[6]);
	}
	return max_diff_info[0];
}



// 主程序
int main()
{
	// 生成随机数
	printf("生成随机数\n");
	generate_matrix();
	printf("随机数为：\n");
	print_matrix(matrix, M, N, PRINT_LIMIT);

	// custom
	printf("使用自定义函数进行FFT\n");
	fft_custom();
	printf("结果为：\n");
	print_matrix_complex(matrix_fft_custom, M, N, PRINT_LIMIT);

	// fftw3
	printf("使用 fftw3 进行FFT\n");
	fft_fftw3();
	printf("压缩形式为：\n");
	print_matrix(matrix_fft_fftw3, M, 2 * (N / 2 + 1), PRINT_LIMIT);
	fft_fftw3_transform();
	printf("结果为：\n");
	print_matrix_complex(matrix_fft_fftw3_trans, M, N, PRINT_LIMIT);

	// mkl
	printf("使用 oneMKL 进行FFT\n");
	fft_mkl();
	printf("压缩形式为：\n");
	print_matrix_complex(matrix_fft_mkl, M, N, PRINT_LIMIT);
	fft_mkl_transform();
	printf("结果为：\n");
	print_matrix_complex(matrix_fft_mkl_trans, M, N, PRINT_LIMIT);

	// 验证结果
	printf("\n比较 matrix_fft_custom 与 matrix_fft_fftw3_trans\n");
	float error_12 = verify(matrix_fft_custom, matrix_fft_fftw3_trans);
	printf((error_12 <= EPSILON)?"结果正确\n": "结果不正确\n");
	
	printf("\n比较 matrix_fft_custom 与 matrix_fft_mkl_trans\n");
	float error_13 = verify(matrix_fft_custom, matrix_fft_mkl_trans);
	printf((error_13<= EPSILON) ? "结果正确\n" : "结果不正确\n");

	printf("\n比较 matrix_fft_fftw3_trans 与 matrix_fft_mkl_trans\n");
	float error_23 = verify(matrix_fft_fftw3_trans, matrix_fft_mkl_trans);
	printf((error_23 <= EPSILON) ? "结果正确\n" : "结果不正确\n");

	if (!RUN_REPEAT)
		return 0;

	// 耗时对比
	printf("\n开始耗时对比，矩阵为 %d * %d，共计 %d 轮\n", M, N, REPEAT);
	auto total_custom = chrono::duration_cast<chrono::nanoseconds>(chrono::nanoseconds::zero());
	auto total_fftw3 = chrono::duration_cast<chrono::nanoseconds>(chrono::nanoseconds::zero());
	auto total_mkl = chrono::duration_cast<chrono::nanoseconds>(chrono::nanoseconds::zero());

	for (int i = 0; i < REPEAT; i++)
	{
		generate_matrix(i);

		if (i % 100 == 0)
		{
			printf("第 %d 次运行\n", i + 1);
			auto start_custom = chrono::high_resolution_clock::now();
			fft_custom();
			auto end_custom = chrono::high_resolution_clock::now();
			total_custom += chrono::duration_cast<chrono::nanoseconds>(end_custom - start_custom);
		}

		auto start_fftw3 = chrono::high_resolution_clock::now();
		fft_fftw3();
		auto end_fftw3 = chrono::high_resolution_clock::now();
		total_fftw3 += chrono::duration_cast<chrono::nanoseconds>(end_fftw3 - start_fftw3);

		auto start_mkl = chrono::high_resolution_clock::now();
		fft_mkl();
		auto end_mkl = chrono::high_resolution_clock::now();
		total_mkl += chrono::duration_cast<chrono::nanoseconds>(end_mkl - start_mkl);

		// 我们不考虑transform后处理的时间	
	}

	printf("自定义函数平均耗时 %.3e s\n", (double)total_custom.count() * 100 / REPEAT  /1e9);
	printf("fftw3 平均耗时 %.3e s\n", (double)total_fftw3.count() / REPEAT / 1e9);
	printf("fftw3 加速比 %.3f\n", (double)total_custom.count() * 100 /total_fftw3.count());
	printf("oneMKL 平均耗时 %.3e s\n", (double)total_mkl.count() / REPEAT / 1e9);
	printf("oneMKL 加速比 %.3f\n", (double)total_custom.count() * 100 / total_mkl.count());
	printf("oneMKL 与 fftw3 加速比 %.3f\n", (double)total_fftw3.count() / total_mkl.count());

	// 将以上数据写入‘readme.md’最后，每个数据三位小数
	// | M | N | 自定义函数 | fftw3 (加速比) | oneMKL (加速比) | fftw3/oneMKL耗时比 |
	FILE* readme = fopen("readme.md", "a");
	fprintf(readme, "| %d | %d | %.3e s | %.3e s (%.3f x) | %.3e s (%.3f x) | %.3f |\n", M,
		N,
		(double)total_custom.count() * 100 / REPEAT  / 1e9,
		(double)total_fftw3.count() / REPEAT / 1e9,
		(double)total_custom.count() * 100 / total_fftw3.count(),
		(double)total_mkl.count() / REPEAT / 1e9,
		(double)total_custom.count() * 100 / total_mkl.count(),
		(double)total_fftw3.count() / total_mkl.count());

	return 0;
}