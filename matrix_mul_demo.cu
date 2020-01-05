#include <stdio.h>
#include <unistd.h>

#include "MatMul.h"
#include "matrix.h"

#define ROW 1024
#define COL 1024

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool finish = false;

typedef WFThreadTask<MatrixIn, MatrixOut> MatMulSyncTask;

void check_sync_result(MatMulSyncTask *task)
{
	fprintf(stderr, "checking in ThreadTask callback.\n");
	MatrixIn *in = task->get_input();
	MatrixOut *out = task->get_output();
	matrix_check(in, out);
    pthread_mutex_lock(&mutex);
    finish = true;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
}

typedef CudaAsyncTask<MatrixIn, MatrixOut> MatMulAsyncTask;

void check_async_result(MatMulAsyncTask *task)
{
	fprintf(stderr, "checking in AsyncTask callback.\n");
	MatrixIn *in = task->get_input();
	MatrixOut *out = task->get_output();
	matrix_check(in, out);
    pthread_mutex_lock(&mutex);
    finish = true;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
}

int main()
{	
    dim3 thread_per_block(16, 16);
	dim3 block_num((COL + thread_per_block.x - 1) / thread_per_block.x,
				   (ROW + thread_per_block.y - 1) / thread_per_block.y);

	MatrixIn *in;
	MatrixOut *out;
/*
	// test thread sync task
	std::string queue_name = "cuda_matrix_mul";
	MatMulSyncTask *sync_task = CUDATaskFactory::create_matmul_sync_task<MatrixIn, MatrixOut>
									(queue_name, block_num, thread_per_block, check_sync_result);

	in = sync_task->get_input();
	out = sync_task->get_output();
	in->init(ROW, COL);
	out->init(ROW, COL);
	in->a = (int *)malloc(sizeof(int) * ROW * COL);
	in->b = (int *)malloc(sizeof(int) * ROW * COL);
	out->c = (int *)malloc(sizeof(int) * ROW * COL);

	init_random(in);

	fprintf(stderr, "start sync thread task\n");
	sync_task->start();

    pthread_mutex_lock(&mutex);
    while (!finish)
        pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);
*/
	// test async task
	MatMulAsyncTask *async_task = CUDATaskFactory::create_matmul_async_task<MatrixIn, MatrixOut>
										(block_num, thread_per_block, check_async_result);

	in = async_task->get_input();
	out = async_task->get_output();
	in->init(ROW, COL);
	out->init(ROW, COL);
	in->a = (int *)malloc(sizeof(int) * ROW * COL);
	in->b = (int *)malloc(sizeof(int) * ROW * COL);
	out->c = (int *)malloc(sizeof(int) * ROW * COL);

	init_random(in);

	fprintf(stderr, "start async request task\n");
	async_task->start();

    pthread_mutex_lock(&mutex);
    while (!finish)
        pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);

	return 0;
}

