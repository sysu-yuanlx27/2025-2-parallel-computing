#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

printf 'program,threads,schedule,seconds\n'
for threads in 1 2 4 8; do
  for schedule in static dynamic; do
    seconds=$(./heated_plate_pthreads "$threads" "$schedule" | awk '/Wallclock time/ {print $4}')
    printf 'pthread,%s,%s,%s\n' "$threads" "$schedule" "$seconds"
  done
done

for threads in 1 2 4 8; do
  seconds=$(OMP_NUM_THREADS="$threads" ./heated_plate_openmp | awk '/Wallclock time/ {print $4}')
  printf 'openmp,%s,default,%s\n' "$threads" "$seconds"
done
