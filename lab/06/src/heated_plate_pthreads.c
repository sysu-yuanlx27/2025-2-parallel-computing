#define _POSIX_C_SOURCE 199309L
#include "parallel_for.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int m;
    int n;
    double *u;
    double *w;
    double mean;
    double *row_diffs;
} plate_args_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t idx(const plate_args_t *args, int i, int j) {
    return (size_t)i * (size_t)args->n + (size_t)j;
}

static void *init_side_boundaries(int i, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    args->w[idx(args, i, 0)] = 100.0;
    args->w[idx(args, i, args->n - 1)] = 100.0;
    return NULL;
}

static void *init_top_bottom_boundaries(int j, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    args->w[idx(args, args->m - 1, j)] = 100.0;
    args->w[idx(args, 0, j)] = 0.0;
    return NULL;
}

static void *init_interior_row(int i, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    for (int j = 1; j < args->n - 1; ++j) args->w[idx(args, i, j)] = args->mean;
    return NULL;
}

static void *copy_row(int i, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    memcpy(&args->u[idx(args, i, 0)], &args->w[idx(args, i, 0)], (size_t)args->n * sizeof(double));
    return NULL;
}

static void *update_row(int i, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    for (int j = 1; j < args->n - 1; ++j) {
        args->w[idx(args, i, j)] =
            (args->u[idx(args, i - 1, j)] + args->u[idx(args, i + 1, j)] +
             args->u[idx(args, i, j - 1)] + args->u[idx(args, i, j + 1)]) / 4.0;
    }
    return NULL;
}

static void *diff_row(int i, void *raw_args) {
    plate_args_t *args = (plate_args_t *)raw_args;
    double local_max = 0.0;
    for (int j = 1; j < args->n - 1; ++j) {
        const double delta = fabs(args->w[idx(args, i, j)] - args->u[idx(args, i, j)]);
        if (delta > local_max) local_max = delta;
    }
    args->row_diffs[i] = local_max;
    return NULL;
}

static pf_schedule_t parse_schedule(const char *text) {
    return strcmp(text, "dynamic") == 0 ? PF_DYNAMIC : PF_STATIC;
}

int main(int argc, char **argv) {
    const int threads = argc > 1 ? atoi(argv[1]) : 4;
    const pf_schedule_t schedule = argc > 2 ? parse_schedule(argv[2]) : PF_STATIC;
    const int m = argc > 3 ? atoi(argv[3]) : 500;
    const int n = argc > 4 ? atoi(argv[4]) : 500;
    const double epsilon = argc > 5 ? atof(argv[5]) : 0.001;
    if (threads <= 0 || m < 3 || n < 3 || epsilon <= 0.0) {
        fprintf(stderr, "Usage: %s [threads] [static|dynamic] [m] [n] [epsilon]\n", argv[0]);
        return 1;
    }

    const size_t cells = (size_t)m * (size_t)n;
    double *u = (double *)calloc(cells, sizeof(*u));
    double *w = (double *)calloc(cells, sizeof(*w));
    double *row_diffs = (double *)calloc((size_t)m, sizeof(*row_diffs));
    if (!u || !w || !row_diffs) {
        fprintf(stderr, "allocation failed\n");
        free(u); free(w); free(row_diffs);
        return 1;
    }

    plate_args_t args = {m, n, u, w, 0.0, row_diffs};
    if (parallel_for_schedule(1, m - 1, 1, init_side_boundaries, &args, threads, schedule, 1) < 0 ||
        parallel_for_schedule(0, n, 1, init_top_bottom_boundaries, &args, threads, schedule, 1) < 0) {
        fprintf(stderr, "parallel_for failed during boundary initialization\n");
        return 1;
    }

    double mean = 0.0;
    for (int i = 1; i < m - 1; ++i) mean += w[idx(&args, i, 0)] + w[idx(&args, i, n - 1)];
    for (int j = 0; j < n; ++j) mean += w[idx(&args, m - 1, j)] + w[idx(&args, 0, j)];
    args.mean = mean / (double)(2 * m + 2 * n - 4);

    if (parallel_for_schedule(1, m - 1, 1, init_interior_row, &args, threads, schedule, 1) < 0) {
        fprintf(stderr, "parallel_for failed during initialization\n");
        return 1;
    }

    printf("HEATED_PLATE_PTHREADS\n");
    printf("grid=%d x %d threads=%d schedule=%s epsilon=%g mean=%.6f\n",
           m, n, threads, pf_schedule_name(schedule), epsilon, args.mean);
    printf(" Iteration  Change\n\n");

    int iterations = 0;
    int iterations_print = 1;
    double diff = epsilon;
    const double begin = now_seconds();
    while (epsilon <= diff) {
        if (parallel_for_schedule(0, m, 1, copy_row, &args, threads, schedule, 1) < 0 ||
            parallel_for_schedule(1, m - 1, 1, update_row, &args, threads, schedule, 1) < 0 ||
            parallel_for_schedule(1, m - 1, 1, diff_row, &args, threads, schedule, 1) < 0) {
            fprintf(stderr, "parallel_for failed during iteration\n");
            return 1;
        }

        diff = 0.0;
        for (int i = 1; i < m - 1; ++i) {
            if (row_diffs[i] > diff) diff = row_diffs[i];
        }
        ++iterations;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print *= 2;
        }
    }
    const double elapsed = now_seconds() - begin;

    printf("\n  %8d  %f\n", iterations, diff);
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %.6f\n", elapsed);

    free(u); free(w); free(row_diffs);
    return 0;
}
