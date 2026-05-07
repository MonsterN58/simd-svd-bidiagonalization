#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++}
STD=${STD:--std=c++17}
RESULT_DIR=${RESULT_DIR:-results}
BUILD_DIR=${BUILD_DIR:-build_bonus}
PERF_SIZE=${PERF_SIZE:-1000}
PERF_REPS=${PERF_REPS:-1}

mkdir -p "$RESULT_DIR" "$BUILD_DIR"

cpu_has() {
    local flag=$1
    grep -qiE "(^|[[:space:]])${flag}([[:space:]]|$)" /proc/cpuinfo 2>/dev/null
}

compiler_supports() {
    local flag=$1
    printf 'int main(){return 0;}\n' | "$CXX" $STD "$flag" -x c++ - -o "$BUILD_DIR/flag_test" >/dev/null 2>&1
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

compile() {
    local name=$1
    shift
    echo "[build] $name: $*"
    "$CXX" $STD "$@" bench_bidiag.cpp bidiagonalization.cpp -o "$BUILD_DIR/bench_$name"
}

run_csv() {
    local name=$1
    local csv=$2
    echo "[run] $name"
    "$BUILD_DIR/bench_$name" "$name" > "$RESULT_DIR/$name.csv"
    if [[ ! -f "$csv" ]]; then
        cp "$RESULT_DIR/$name.csv" "$csv"
    else
        tail -n +2 "$RESULT_DIR/$name.csv" >> "$csv"
    fi
}

echo "== SIMD width experiments =="
rm -f "$RESULT_DIR/simd_width.csv"
compile scalar_o2 -O2 -DDISABLE_MANUAL_SIMD
if compiler_supports -msse2; then
    compile sse2_o2 -O2 -msse2
else
    echo "[info] compiler does not accept -msse2; build sse2_o2 with plain -O2."
    compile sse2_o2 -O2
fi
run_csv scalar_o2 "$RESULT_DIR/simd_width.csv"
run_csv sse2_o2 "$RESULT_DIR/simd_width.csv"

if cpu_has avx && compiler_supports -mavx; then
    compile avx_o2 -O2 -mavx
    run_csv avx_o2 "$RESULT_DIR/simd_width.csv"
else
    echo "[skip] AVX was not detected or compiler does not accept -mavx; skip 4-way runtime test."
fi

if cpu_has avx512f && compiler_supports -mavx512f; then
    compile avx512_o2 -O2 -mavx512f
    run_csv avx512_o2 "$RESULT_DIR/simd_width.csv"
else
    echo "[skip] AVX-512 was not detected or compiler does not accept -mavx512f; skip 8-way runtime test."
fi

echo "== Compiler optimization experiments =="
rm -f "$RESULT_DIR/compiler_opts.csv"
for opt in O0 O1 O2; do
    opt_lower=$(lower "$opt")
    compile scalar_${opt_lower} "-$opt" -DDISABLE_MANUAL_SIMD
    run_csv scalar_${opt_lower} "$RESULT_DIR/compiler_opts.csv"
    if cpu_has avx && compiler_supports -mavx; then
        compile avx_${opt_lower} "-$opt" -mavx
        run_csv avx_${opt_lower} "$RESULT_DIR/compiler_opts.csv"
    else
        echo "[skip] AVX was not detected or compiler does not accept -mavx; skip avx_${opt_lower}."
    fi
done

echo "== Alignment microbenchmark =="
if cpu_has avx && compiler_supports -mavx; then
    "$CXX" $STD -O2 -mavx bench_alignment.cpp -o "$BUILD_DIR/bench_alignment"
    "$BUILD_DIR/bench_alignment" > "$RESULT_DIR/alignment.csv"
else
    echo "[skip] AVX was not detected or compiler does not accept -mavx; skip alignment microbenchmark."
fi

echo "== perf profiling =="
if command -v perf >/dev/null 2>&1; then
    perf stat -r 3 -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
        "$BUILD_DIR/bench_scalar_o2" scalar_profile "$PERF_SIZE" "$PERF_REPS" \
        > "$RESULT_DIR/perf_scalar_stdout.txt" 2> "$RESULT_DIR/perf_scalar_stat.txt" || true

    if [[ -x "$BUILD_DIR/bench_avx_o2" ]]; then
        perf stat -r 3 -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
            "$BUILD_DIR/bench_avx_o2" avx_profile "$PERF_SIZE" "$PERF_REPS" \
            > "$RESULT_DIR/perf_avx_stdout.txt" 2> "$RESULT_DIR/perf_avx_stat.txt" || true

        perf record -g -o "$RESULT_DIR/perf_avx.data" -- \
            "$BUILD_DIR/bench_avx_o2" avx_profile "$PERF_SIZE" "$PERF_REPS" >/dev/null 2>&1 || true
        perf report --stdio -i "$RESULT_DIR/perf_avx.data" > "$RESULT_DIR/perf_avx_report.txt" 2>/dev/null || true
    fi
else
    echo "[skip] perf is not available on this machine."
fi

echo "Done. Results are under $RESULT_DIR/."
