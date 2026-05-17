#include "parallel_for.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef struct {
    int start;
    int inc;
    int total_iterations;
    int chunk_size;
    void *(*functor)(int, void *);
    void *arg;
    pf_schedule_t schedule;
    atomic_int next_iteration;
} parallel_task_t;

typedef struct {
    parallel_task_t *task;
    int begin_iteration;
    int end_iteration;
} worker_arg_t;

static int iteration_count(int start, int end, int inc) {
    if (inc > 0) {
        if (start >= end) return 0;
        return (end - start + inc - 1) / inc;
    }
    if (inc < 0) {
        if (start <= end) return 0;
        const int step = -inc;
        return (start - end + step - 1) / step;
    }
    return -1;
}

static void *worker_main(void *raw_arg) {
    worker_arg_t *worker = (worker_arg_t *)raw_arg;
    parallel_task_t *task = worker->task;

    if (task->schedule == PF_DYNAMIC) {
        for (;;) {
            const int begin = atomic_fetch_add(&task->next_iteration, task->chunk_size);
            if (begin >= task->total_iterations) break;
            int end = begin + task->chunk_size;
            if (end > task->total_iterations) end = task->total_iterations;
            for (int iter = begin; iter < end; ++iter) {
                task->functor(task->start + iter * task->inc, task->arg);
            }
        }
        return NULL;
    }

    for (int iter = worker->begin_iteration; iter < worker->end_iteration; ++iter) {
        task->functor(task->start + iter * task->inc, task->arg);
    }
    return NULL;
}

int parallel_for_schedule(int start,
                          int end,
                          int inc,
                          void *(*functor)(int, void *),
                          void *arg,
                          int num_threads,
                          pf_schedule_t schedule,
                          int chunk_size) {
    if (functor == NULL || inc == 0 || num_threads <= 0) return -1;
    if (schedule != PF_STATIC && schedule != PF_DYNAMIC) return -1;
    if (chunk_size <= 0) chunk_size = 1;

    const int total_iterations = iteration_count(start, end, inc);
    if (total_iterations < 0) return -1;
    if (total_iterations == 0) return 0;

    const int actual_threads = num_threads < total_iterations ? num_threads : total_iterations;
    pthread_t *threads = (pthread_t *)calloc((size_t)actual_threads, sizeof(*threads));
    worker_arg_t *worker_args = (worker_arg_t *)calloc((size_t)actual_threads, sizeof(*worker_args));
    if (threads == NULL || worker_args == NULL) {
        free(threads);
        free(worker_args);
        return -1;
    }

    parallel_task_t task;
    task.start = start;
    task.inc = inc;
    task.total_iterations = total_iterations;
    task.chunk_size = chunk_size;
    task.functor = functor;
    task.arg = arg;
    task.schedule = schedule;
    atomic_init(&task.next_iteration, 0);

    int created = 0;
    for (int tid = 0; tid < actual_threads; ++tid) {
        worker_arg_t *worker = &worker_args[tid];
        worker->task = &task;

        if (schedule == PF_STATIC) {
            const int base = total_iterations / actual_threads;
            const int extra = total_iterations % actual_threads;
            worker->begin_iteration = tid * base + (tid < extra ? tid : extra);
            worker->end_iteration = worker->begin_iteration + base + (tid < extra ? 1 : 0);
        }

        if (pthread_create(&threads[tid], NULL, worker_main, worker) != 0) {
            break;
        }
        ++created;
    }

    int status = created == actual_threads ? actual_threads : -1;
    for (int tid = 0; tid < created; ++tid) {
        if (pthread_join(threads[tid], NULL) != 0) status = -1;
    }

    free(threads);
    free(worker_args);
    return status;
}

int parallel_for(int start,
                 int end,
                 int inc,
                 void *(*functor)(int, void *),
                 void *arg,
                 int num_threads) {
    return parallel_for_schedule(start, end, inc, functor, arg, num_threads, PF_STATIC, 1);
}

const char *pf_schedule_name(pf_schedule_t schedule) {
    return schedule == PF_DYNAMIC ? "dynamic" : "static";
}
