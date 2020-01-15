#ifndef __CUDA_TASK_MATRIX_H__
#define __CUDA_TASK_MATRIX_H__

#include <random>

/*
class MatrixIn;
class MatrixOut;
typedef CUDATask<MatrixIn, MatrixOut> CudaMMTask;
*/
template<typename T>
class MatrixIn
{
public:
	MatrixIn()
	{
		this->a = nullptr;
		this->b = nullptr;
	}
	~MatrixIn() { }

	void init(int row, int col)
	{
		this->row = row;
		this->col = col;
	}
	
	T *a, *b;
	int row, col;
};

template<typename T>
class MatrixOut
{
public:
	MatrixOut()
	{
		this->c = nullptr;
	}
	~MatrixOut() { }

	void init(int row, int col)
	{
		this->row = row;
		this->col = col;
	}

	T *c;
	int row, col;
};

void matrix_mul_cpu(int *a, int* b, int* c, int width)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int sum = 0;
			for (int k = 0; k < width; k++)
			{
				int tmp_a = a[i * width + k];
				int tmp_b = b[k * width + j];
				sum += tmp_a * tmp_b;
			}
			c[i * width + j] = sum;
		}
	}
}

inline void matrix_check(MatrixIn<int> *in, MatrixOut<int> *out)
{
	bool correct = true;
	int col = in->col;
	int row = in->row;
	int size = sizeof(int) * row * col;
	int *a = in->a;
	int *b = in->b;
	int *c = out->c;
	int *cref = (int *)malloc(size);

	fprintf(stderr, "c[0]=%d\n", c[0]);
	matrix_mul_cpu(a, b, cref, col);
	fprintf(stderr, "cref[0]=%d\n", cref[0]);

	for (int i = 0; i < row * col; i++)
	{
		if (c[i] != cref[i])
		{
			correct = false;
			fprintf(stderr, "c[%d]=%d but cref[%d]=%d\n",
					i, c[i], i, cref[i]);
			break;
		}
	}
	fprintf(stderr, "%s result!\n", correct ? "Correct": "Wrong");
}

inline void init_random(MatrixIn<int> *in)
{
	std::default_random_engine random_engine;
	std::uniform_int_distribution<int> urandom_gen(0, 1000);

	int col = in->col;
	int row = in->row;

	for (int i = 0; i < row * col; i++)
	{
		in->a[i] = urandom_gen(random_engine);
		in->b[i] = urandom_gen(random_engine);
	}
}

#endif
