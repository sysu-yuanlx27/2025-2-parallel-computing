#define _POSIX_C_SOURCE 199309L
#include "parallel_for.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const double *a;
    const double *b;
    double *c;
    int n;
} matmul_args_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *matmul_row(int row, void *raw_args) {
    matmul_args_t *args = (matmul_args_t *)raw_args;
    const int n = args->n;
    for (int k = 0; k < n; ++k) {
        const double aik = args->a[row * n + k];
        for (int col = 0; col < n; ++col) {
            args->c[row * n + col] += aik * args->b[k * n + col];
        }
    }
    return NULL;
}

static void serial_matmul(const double *a, const double *b, double *c, int n) {
    memset(c, 0, (size_t)n * (size_t)n * sizeof(*c));
    for (int row = 0; row < n; ++row) {
        for (int k = 0; k < n; ++k) {
            const double aik = a[row * n + k];
            for (int col = 0; col < n; ++col) {
                c[row * n + col] += aik * b[k * n + col];
            }
        }
    }
}

static pf_schedule_t parse_schedule(const char *text) {
    return strcmp(text, "dynamic") == 0 ? PF_DYNAMIC : PF_STATIC;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 256;
    const int threads = argc > 2 ? atoi(argv[2]) : 4;
    const pf_schedule_t schedule = argc > 3 ? parse_schedule(argv[3]) : PF_STATIC;
    if (n <= 0 || threads <= 0) {
        fprintf(stderr, "Usage: %s [matrix_size] [threads] [static|dynamic]\n", argv[0]);
        return 1;
    }

    const size_t cells = (size_t)n * (size_t)n;
    double *a = (double *)malloc(cells * sizeof(*a));
    double *b = (double *)malloc(cells * sizeof(*b));
    double *parallel_c = (double *)calloc(cells, sizeof(*parallel_c));
    double *serial_c = (double *)malloc(cells * sizeof(*serial_c));
    if (!a || !b || !parallel_c || !serial_c) {
        fprintf(stderr, "allocation failed\n");
        free(a); free(b); free(parallel_c); free(serial_c);
        return 1;
    }

    for (size_t i = 0; i < cells; ++i) {
        a[i] = (double)((i * 17u) % 101u) / 101.0;
        b[i] = (double)((i * 29u) % 103u) / 103.0;
    }

    matmul_args_t args = {a, b, parallel_c, n};
    const double parallel_begin = now_seconds();
    if (parallel_for_schedule(0, n, 1, matmul_row, &args, threads, schedule, 1) < 0) {
        fprintf(stderr, "parallel_for failed\n");
        return 1;
    }
    const double parallel_time = now_seconds() - parallel_begin;

    const double serial_begin = now_seconds();
    serial_matmul(a, b, serial_c, n);
    const double serial_time = now_seconds() - serial_begin;

    double max_error = 0.0;
    for (size_t i = 0; i < cells; ++i) {
        const double error = fabs(parallel_c[i] - serial_c[i]);
        if (error > max_error) max_error = error;
    }

    printf("matrix_size=%d threads=%d schedule=%s\n", n, threads, pf_schedule_name(schedule));
    printf("parallel_time=%.6f s serial_time=%.6f s speedup=%.3f\n",
           parallel_time, serial_time, serial_time / parallel_time);
    printf("max_error=%.12g result=%s\n", max_error, max_error < 1e-9 ? "PASS" : "FAIL");

    free(a); free(b); free(parallel_c); free(serial_c);
    return max_error < 1e-9 ? 0 : 2;
}
