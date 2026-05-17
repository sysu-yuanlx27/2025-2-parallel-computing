#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PF_STATIC = 0,
    PF_DYNAMIC = 1
} pf_schedule_t;

int parallel_for(int start,
                 int end,
                 int inc,
                 void *(*functor)(int, void *),
                 void *arg,
                 int num_threads);

int parallel_for_schedule(int start,
                          int end,
                          int inc,
                          void *(*functor)(int, void *),
                          void *arg,
                          int num_threads,
                          pf_schedule_t schedule,
                          int chunk_size);

const char *pf_schedule_name(pf_schedule_t schedule);

#ifdef __cplusplus
}
#endif

#endif
