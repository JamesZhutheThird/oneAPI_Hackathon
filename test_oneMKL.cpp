#include <iostream>
#include <mkl.h>
#include <chrono>
#include<stdio.h>
#include<stdlib.h>
#include <string>

using namespace std;

int main()
{
	double* rand = new double[10];

	VSLStreamStatePtr stream;
	int seed = 111;
	double rang[2] = { -2, 2 };

	auto stutes = vslNewStream(&stream, VSL_BRNG_MT19937, seed);
	stutes = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 10, rand, rang[0], rang[1]);
	stutes = vslDeleteStream(&stream);

	cout << "���� 10��{-2,2} ��Χ�ھ��ȷֲ��������" << endl;
	cout << "�������óɹ�" << endl;
	cout << "���������: " << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << i << " : " << rand[i] << endl;
	}

	return 0;
}