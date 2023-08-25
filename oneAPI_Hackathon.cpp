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
#include <mkl.h>
#include <fftw3.h>
using namespace std;

// 定义宏
#define M 2048
#define N 2048
#define PRINT_LIMIT 10
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define REPEAT 1000

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
complex<float>* matrix_fft_custom = (complex<float>*)mkl_malloc(sizeof(complex<float>) * M * N, 64);/* save matrix after fft_custom */

void fft()
{

}

void fft_custom()
{

}

// 2D Real to complex FFT with fftw3.h
float* matrix_fft_fftw3 = (float*)fftwf_malloc(sizeof(float) * 2 * M * (N / 2 + 1));/* save matrix after fft_fftw3 */
complex<float>* matrix_fft_fftw3_trans = (complex<float>*)mkl_malloc(sizeof(complex<float>) * M * N, 64); /* decode matrix after fft_fftw3 */

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
complex<float>* matrix_fft_mkl = (complex<float>*)mkl_malloc(sizeof(complex<float>) * M * N, 64); /* save matrix after fft_mkl */
complex<float>* matrix_fft_mkl_trans = (complex<float>*)mkl_malloc(sizeof(complex<float>) * M * N, 64); /* decode matrix after fft_mkl */

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
float verify(complex<float>* mat_1, complex<float>* mat_2, float epsilon = 1e-5)
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

void print_matrix_complex(complex<float>* matrix_to_print, int m, int n, int limit = 0)
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

// 主程序
int main()
{
	// 生成随机数
	printf("生成随机数\n");
	generate_matrix();
	printf("随机数为：\n");
	print_matrix(matrix, M, N, PRINT_LIMIT);

	// fftw3
	printf("使用 fftw3 进行FFT\n");
	fft_fftw3();
	print_matrix(matrix_fft_fftw3, M, 2 * (N / 2 + 1), PRINT_LIMIT);
	fft_fftw3_transform();
	print_matrix_complex(matrix_fft_fftw3_trans, M, N, PRINT_LIMIT);

	// mkl
	printf("使用 oneMKL 进行FFT\n");
	fft_mkl();
	print_matrix_complex(matrix_fft_mkl, M, N, PRINT_LIMIT);
	fft_mkl_transform();
	print_matrix_complex(matrix_fft_mkl_trans, M, N, PRINT_LIMIT);

	// 验证结果
	printf("最大误差为%f\n", verify(matrix_fft_fftw3_trans, matrix_fft_mkl_trans));
	if (verify(matrix_fft_fftw3_trans, matrix_fft_mkl_trans) < EPSILON)
		printf("结果正确\n");
	else
		printf("结果不正确\n");

	// 耗时对比
	printf("开始耗时对比，矩阵为 %d * %d，共计 %d 轮\n", M, N, REPEAT);
	auto total_fftw3 = chrono::duration_cast<chrono::milliseconds>(chrono::milliseconds::zero());
	auto total_mkl = chrono::duration_cast<chrono::milliseconds>(chrono::milliseconds::zero());

	for (int i = 0; i < REPEAT; i++)
	{
		if (i % (int)ceil((float)REPEAT / 10) == 0)
			printf("第 %d 次运行\n", i + 1);

		generate_matrix(i);
		auto start_fftw3 = chrono::high_resolution_clock::now();
		fft_fftw3();
		auto end_fftw3 = chrono::high_resolution_clock::now();
		total_fftw3 += chrono::duration_cast<chrono::milliseconds>(end_fftw3 - start_fftw3);

		auto start_mkl = chrono::high_resolution_clock::now();
		fft_mkl();
		auto end_mkl = chrono::high_resolution_clock::now();
		total_mkl += chrono::duration_cast<chrono::milliseconds>(end_mkl - start_mkl);

		// 我们不考虑transform后处理的时间	
	}

	printf("fftw3 平均耗时 %.3f ms\n", (float)total_fftw3.count() / REPEAT);
	printf("mkl 平均耗时 %.3f ms\n", (float)total_mkl.count() / REPEAT);
	printf("加速比 %f\n", (float)total_fftw3.count() / total_mkl.count());

	return 0;
}