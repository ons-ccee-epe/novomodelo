#!/usr/bin/env bash
#
# check-release-kernel.sh — release probe for the cut-selection dgemm kernel.
#
# Reads built x86_64 Linux artifacts and fails when one carries a packed fused
# multiply-add instruction. A flag that reaches the build of what ships (a
# RUSTFLAGS in a release workflow, a maturin argument, a cargo-dist setting,
# CFLAGS/CXXFLAGS reaching the cc/cmake builds of the vendored solvers) changes
# the kernel without touching a file the text guard reads; this probe sees the
# instructions themselves.
#
# Usage: check-release-kernel.sh <artifact>...
#        check-release-kernel.sh --verify
#
# Each <artifact> is an ELF file, a .tar.xz/.tar.gz archive or a .whl. Archives
# and wheels are extracted under $TMPDIR and every ELF file inside is probed.
# A probed file is refused unless `objdump -f` reports elf64-x86-64 and its
# disassembly holds at least one instruction. The release binaries are
# stripped, so the count covers the whole disassembly, linked C/C++ code
# included; scalar vfmadd...sd forms are not counted.
#
# `--verify` assembles one object holding vfmadd231pd, one holding only
# vmulpd/vaddpd, a 32-bit one and one without code, packs them into the three container kinds,
# and checks that the probe fails, passes and refuses as it must. Its files live under $TMPDIR
# and are always removed.
#
# Needs objdump, tar, python3 and od; --verify also needs as. GNU grep (for \b).
#
# Exit codes:
#   0 — Pass: every probed file holds 0 packed FMA instructions (or --verify
#       succeeded).
#   1 — Fail: a probed file holds packed FMA instructions (or --verify saw a
#       wrong outcome).
#   2 — Refused: bad invocation, a missing argument, an archive or wheel with
#       no ELF file inside, or a file that is not x86_64 ELF or disassembles to
#       nothing. A refusal takes precedence over a failure.

set -euo pipefail

readonly PACKED_FMA='\bvf(n)?m(add|sub|addsub|subadd)(132|213|231)p[sd]\b'
readonly SCRIPT_PATH="${BASH_SOURCE[0]}"

WORK_DIR=""
cleanup() {
    if [[ -n "$WORK_DIR" ]]; then
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

work_dir() {
    if [[ -z "$WORK_DIR" ]]; then
        WORK_DIR="$(mktemp -d -t release_kernel.XXXXXX)"
    fi
}

is_elf() {
    [[ "$(od -An -tx1 -N4 -- "$1" 2>/dev/null | tr -d ' \n')" == "7f454c46" ]]
}

# probe_file <path> <label>
#
# Prints one OK/FAIL/REFUSED line for <path> under the name <label>.
# Returns 0 (OK), 1 (FAIL) or 2 (REFUSED).
probe_file() {
    local path="$1" label="$2"
    local format
    format="$(objdump -f -- "$path" 2>/dev/null)" || format=""
    if [[ "$format" != *"file format elf64-x86-64"* ]]; then
        echo "REFUSED: ${label}: not an x86_64 ELF file"
        return 2
    fi

    work_dir
    local disasm="${WORK_DIR}/disasm.txt"
    if ! objdump -d --no-show-raw-insn -- "$path" >"$disasm" 2>/dev/null; then
        echo "REFUSED: ${label}: objdump -d failed"
        return 2
    fi
    if ! grep -qE '^ *[0-9a-f]+:'$'\t' "$disasm"; then
        echo "REFUSED: ${label}: empty disassembly"
        return 2
    fi

    local count
    count="$(grep -cE -e "$PACKED_FMA" "$disasm")" || true
    if [[ "$count" -eq 0 ]]; then
        echo "OK: ${label}: 0 packed FMA instructions"
        return 0
    fi
    echo "FAIL: ${label}: ${count} packed FMA instructions"
    grep -E -e "$PACKED_FMA" "$disasm" | head -n 3 | sed 's/^/    /'
    return 1
}

# extract <artifact> <dir>
extract() {
    local artifact="$1" dir="$2"
    case "$artifact" in
        *.tar.xz) tar --no-same-owner -xJf "$artifact" -C "$dir" ;;
        *.tar.gz) tar --no-same-owner -xzf "$artifact" -C "$dir" ;;
        *.whl) python3 -m zipfile -e "$artifact" "$dir" ;;
    esac
}

# probe_artifact <artifact>
#
# Returns 0, 1 or 2 like probe_file, over every ELF file the artifact holds.
probe_artifact() {
    local artifact="$1"
    if [[ ! -f "$artifact" ]]; then
        echo "REFUSED: ${artifact}: no such file"
        return 2
    fi
    case "$artifact" in
        *.tar.xz | *.tar.gz | *.whl) ;;
        *)
            probe_file "$artifact" "$artifact"
            return
            ;;
    esac

    work_dir
    local dir
    dir="$(mktemp -d -p "$WORK_DIR" extract.XXXXXX)"
    if ! extract "$artifact" "$dir"; then
        echo "REFUSED: ${artifact}: extraction failed"
        return 2
    fi

    local status=0 found=0 file rc
    while IFS= read -r -d '' file; do
        is_elf "$file" || continue
        found=1
        rc=0
        probe_file "$file" "${artifact}!${file#"${dir}/"}" || rc=$?
        if [[ $rc -gt $status ]]; then
            status=$rc
        fi
    done < <(find "$dir" -type f -print0 | sort -z)
    rm -rf "$dir"

    if [[ $found -eq 0 ]]; then
        echo "REFUSED: ${artifact}: no ELF file inside"
        return 2
    fi
    return "$status"
}

probe_all() {
    local status=0 artifact rc
    for artifact in "$@"; do
        rc=0
        probe_artifact "$artifact" || rc=$?
        if [[ $rc -gt $status ]]; then
            status=$rc
        fi
    done
    return "$status"
}

# expect <label> <want exit> <want line prefix> <artifact>
#
# Runs the probe as a separate process and records a failure in
# VERIFY_FAILURES unless it exits <want exit> and its output holds a line
# starting with <want line prefix>.
expect() {
    local label="$1" want_rc="$2" want_prefix="$3" artifact="$4"
    local out rc=0
    out="$(bash "$SCRIPT_PATH" "$artifact" 2>&1)" || rc=$?
    if [[ $rc -ne $want_rc ]]; then
        VERIFY_FAILURES+=("${label}: exit ${rc}, expected ${want_rc}")
    elif ! grep -q "^${want_prefix}" <<<"$out"; then
        VERIFY_FAILURES+=("${label}: no line starting with '${want_prefix}'")
    fi
}

run_verify() {
    local d
    work_dir
    d="${WORK_DIR}/verify"
    mkdir -p "$d/fused" "$d/plain" "$d/text"

    printf '\t.text\n\tvfmadd231pd %%ymm2, %%ymm1, %%ymm0\n\tret\n' >"$d/fused.s"
    printf '\t.text\n\tvmulpd %%ymm2, %%ymm1, %%ymm3\n\tvaddpd %%ymm3, %%ymm0, %%ymm0\n\tret\n' \
        >"$d/plain.s"
    as --64 -o "$d/fused/kernel.o" "$d/fused.s"
    as --64 -o "$d/plain/kernel.o" "$d/plain.s"
    as --32 -o "$d/i386.o" "$d/plain.s"
    printf '\t.data\n\t.quad 1\n' >"$d/data.s"
    as --64 -o "$d/data.o" "$d/data.s"
    echo "not an ELF file" >"$d/text/README.txt"

    tar -czf "$d/fused.tar.gz" -C "$d/fused" kernel.o
    tar -cJf "$d/text.tar.xz" -C "$d/text" README.txt
    (cd "$d/plain" && python3 -m zipfile -c "$d/plain.whl" kernel.o)

    VERIFY_FAILURES=()
    expect "fused object" 1 "FAIL: " "$d/fused/kernel.o"
    expect "plain object" 0 "OK: " "$d/plain/kernel.o"
    expect "non-ELF file" 2 "REFUSED: " "$d/text/README.txt"
    expect "32-bit ELF object" 2 "REFUSED: " "$d/i386.o"
    expect "object without code" 2 "REFUSED: " "$d/data.o"
    expect "fused .tar.gz" 1 "FAIL: " "$d/fused.tar.gz"
    expect "plain .whl" 0 "OK: " "$d/plain.whl"
    expect ".tar.xz without ELF" 2 "REFUSED: " "$d/text.tar.xz"
    expect "missing file" 2 "REFUSED: " "$d/absent.tar.xz"

    if [[ ${#VERIFY_FAILURES[@]} -gt 0 ]]; then
        echo "FAIL: --verify:"
        local m
        for m in "${VERIFY_FAILURES[@]}"; do
            echo "  - $m"
        done
        return 1
    fi
    echo "OK: --verify passed (fail, pass and refuse outcomes over ELF, .tar.gz, .tar.xz and .whl)."
    return 0
}

usage() {
    echo "Usage: $0 <artifact>... | --verify" >&2
    echo "" >&2
    echo "  <artifact>  An ELF file, a .tar.xz/.tar.gz archive or a .whl to probe." >&2
    echo "  --verify    Run the self-test against assembled fixtures." >&2
}

for tool in objdump tar python3 od; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "REFUSED: ${tool} not found" >&2
        exit 2
    fi
done

case "${1:-}" in
    --verify)
        if [[ $# -ne 1 ]]; then
            usage
            exit 2
        fi
        if ! command -v as >/dev/null 2>&1; then
            echo "REFUSED: as not found" >&2
            exit 2
        fi
        if run_verify; then
            exit 0
        else
            exit 1
        fi
        ;;
    '' | -*)
        usage
        exit 2
        ;;
    *)
        rc=0
        probe_all "$@" || rc=$?
        exit "$rc"
        ;;
esac
