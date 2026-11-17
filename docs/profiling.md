# Profiling Novomodelo on Ubuntu 24.04

This guide takes you from a fresh Ubuntu 24.04 machine to CPU profiles, heap
profiles and memory-error reports of the `novomodelo` CLI. It covers:

| Question                                      | Tool                              | Section                                         |
| --------------------------------------------- | --------------------------------- | ----------------------------------------------- |
| Where does the run spend CPU time?            | `perf`, `flamegraph`, `hotspot`   | [CPU profiling](#6-cpu-profiling)               |
| How many instructions does a change cost?     | Valgrind Callgrind, `kcachegrind` | [Callgrind](#callgrind)                         |
| Which code allocates, and how often?          | DHAT, `heaptrack`                 | [Heap profiling](#7-heap-profiling)             |
| Is there a memory error across the C/C++ FFI? | Valgrind Memcheck                 | [Memory errors](#8-memory-errors)               |

**Scope.** The steps are for Ubuntu 24.04 installed on x86_64 hardware. `perf`
reads the CPU's hardware performance counters, which virtual machines and WSL
often do not expose. Multi-rank MPI runs are not covered.

Run every command from the repository root unless a step says otherwise.

## 1. Prerequisites

- The build prerequisites in [CONTRIBUTING.md → Prerequisites](../CONTRIBUTING.md#prerequisites)
  (Rust through `rustup`, a C/C++ compiler, CMake).
- The vendored solver sources: `git submodule update --init --recursive`.
- `sudo` rights for the install and kernel-setting steps.

## 2. Install the tools

```bash
sudo apt install valgrind heaptrack heaptrack-gui hotspot kcachegrind
cargo install flamegraph --locked
```

`cargo install flamegraph` puts two commands in `~/.cargo/bin`: `flamegraph`,
which this guide uses, and `cargo-flamegraph`.

`perf` ships in a package tied to the running kernel. Install the metapackage
for your kernel flavour, so `perf` keeps working after kernel updates:

```bash
if dpkg -s linux-image-generic-hwe-24.04 >/dev/null 2>&1; then
  sudo apt install linux-tools-generic-hwe-24.04   # hardware-enablement kernel
else
  sudo apt install linux-tools-generic             # general-availability kernel
fi
perf --version
```

If `perf --version` prints `WARNING: perf not found for kernel …`, the tools
package for the running kernel is missing; rerun the install above.

## 3. Allow perf to sample your own processes

Ubuntu's kernel sets `kernel.perf_event_paranoid` to `4`, which blocks `perf`
for normal users. Level `2`, the upstream kernel default, lets you profile your
own processes in user space, which is all this guide needs:

```bash
echo 'kernel.perf_event_paranoid = 2' | sudo tee /etc/sysctl.d/60-perf-profiling.conf
sudo sysctl --system
cat /proc/sys/kernel/perf_event_paranoid   # prints 2
```

Check that hardware counters are readable:

```bash
perf stat -e cycles,instructions -- true
```

The output lists cycle and instruction counts. On a hybrid Intel CPU each event
appears twice, as `cpu_core/…` (performance cores) and `cpu_atom/…` (efficiency
cores); the core type a short process did not run on reads `<not counted>`.

## 4. Build the binaries

### The CLI, for every tool except DHAT

```bash
scripts/build-profiling.sh
```

This builds `target/perf/profiling/novomodelo` with the `profiling` profile from
`Cargo.toml`: release optimisation, plus the line-level debug info and symbols
the profilers need to name functions and source lines. In this profile the
vendored HiGHS and CLP solvers are also compiled with line tables, so
`perf annotate` and `hotspot` can show which source lines inside a solver
function the samples fall on.

The script exists because a plain `cargo build --profile profiling` gives a
binary that `perf` cannot unwind:

- The pinned Rust toolchain links with its bundled `rust-lld`. `perf`'s DWARF
  unwinder rejects the segment layout `rust-lld` produces (`address range
  overlaps an existing module`) and truncates every call stack to one or two
  frames. The script links with GNU `ld` instead (`-Clinker-features=-lld`).
- A cargo profile cannot set linker flags, so the script passes the flag
  through `RUSTFLAGS`. A `RUSTFLAGS` change rebuilds everything, so the script
  builds into its own target directory, `target/perf/`, and leaves
  `target/release/` and `target/profiling/` untouched.
- `rustup` chooses the toolchain pinned in `rust-toolchain.toml` from the
  current directory. The script changes to the repository root first, so it
  builds with the pinned toolchain wherever you call it from.

The first run compiles the whole workspace, vendored solvers included; later
runs are incremental. Arguments are passed to `cargo build`, so
`scripts/build-profiling.sh -p novomodelo-sddp --example dhat_baseline` builds an
example instead of the CLI.

To confirm which linker produced a binary:

```bash
readelf -p .comment target/perf/profiling/novomodelo | grep -E 'Linker|rustc'
```

A `Linker: LLD …` line means `rust-lld`; the script's binary shows only the
`rustc version` line.

### The DHAT example

DHAT needs the `dhat-heap` feature, which replaces the global allocator, so it
has its own build; the command is in [DHAT](#dhat-allocations-on-the-training-path).

## 5. Shell setup and workloads

The commands in the next sections use three shell variables:

```bash
BIN=target/perf/profiling/novomodelo
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo "0-$(($(nproc) - 1))")
PERF_RECORD="record -e cycles:u -F 997 --call-graph dwarf,64000 -g -o perf.data"
```

- `PCORES` holds the performance cores on a hybrid Intel CPU and all CPUs
  elsewhere. The commands put `taskset -c "$PCORES"` in front of `perf` or
  `flamegraph`, so the profiler and the run it starts all stay on the
  performance cores. The efficiency cores run at a different speed and would
  blur the profile. Each core type also has its own `cycles` event, and
  `flamegraph` draws only the event of the first sample it reads, so one sample
  taken on an efficiency core can push every other sample out of the graph.
- `PERF_RECORD` holds the `perf record` arguments for every recording in this
  guide. `-e cycles:u` matters: `perf`'s default event samples precisely
  (`cycles:P`), and with precise sampling DWARF unwinding fails for part of the
  samples (the [stack check](#checking-that-the-stacks-unwound) reports them).
  `-F` is the sampling rate in samples per second, and `dwarf,64000` copies up
  to 64 000 bytes of stack per sample for the unwinder. Without `-o`,
  `perf record` writes the recording to standard output whenever that is a
  pipe, for example when you pipe the command into `tee`.

`examples/4ree` runs long enough for `perf` to collect a useful number of
samples. Its results go to `examples/4ree/output/`, which git ignores.

Valgrind-based tools run the program on a simulated CPU and are much slower,
so give them a shortened copy of the case with fewer training iterations:

```bash
cp -r examples/4ree /tmp/4ree-short
python3 - /tmp/4ree-short/config.json <<'EOF'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
for rule in config["training"]["stopping_rules"]:
    if rule["type"] == "iteration_limit":
        rule["limit"] = 32
json.dump(config, open(path, "w"), indent=2)
EOF
```

The CLI's `--threads N` option runs N worker threads; `perf`, `heaptrack` and
Valgrind follow every thread of the process.

## 6. CPU profiling

### Hardware counters

```bash
taskset -c "$PCORES" perf stat \
  -e task-clock,cycles,instructions,cache-misses,branch-misses -- \
  "$BIN" run examples/4ree --quiet
```

`perf stat` counts events without sampling, so it barely slows the run; use it
for before/after timing of a change. Run it several times: the spread between
runs tells you how large a difference must be before it means anything.

On a hybrid CPU, read the `cpu_core/…` lines. With the run pinned to the
performance cores, the `cpu_atom/…` lines read `<not counted>`; counts there
mean part of the run executed on the efficiency cores.

### Flamegraph

```bash
taskset -c "$PCORES" flamegraph -c "$PERF_RECORD" -o flamegraph.svg -- \
  "$BIN" run examples/4ree --quiet
```

This records `perf.data` and writes `flamegraph.svg`, both in the current
directory; git ignores both at the repository root. Open the SVG in a browser:
click a frame to zoom into it, and use Search to highlight a function across
the graph. `-c` replaces `flamegraph`'s own `perf record` arguments, which use
the precise default event.

Every sample stores a copy of the thread's stack, so `perf.data` grows with the
run length and the sampling rate. When `perf` prints `Check IO/CPU overload!`
it could not write samples as fast as it took them and lost some; lower `-F` in
`PERF_RECORD` and record again.

### Checking that the stacks unwound

A profile with broken stacks still renders and can point you at the wrong
code, so check a recording before you draw conclusions from it:

```bash
perf script -v 2>&1 >/dev/null | grep -c 'overlaps an existing module'
```

It should print `0`. If every recording of a binary prints a non-zero count,
either the binary was linked with `rust-lld` (rebuild it with
`scripts/build-profiling.sh`) or the recordings used a precise event (record
with `PERF_RECORD`). With both fixes in place, a recording occasionally still
prints a non-zero count and the next recording of the same binary prints `0`;
discard that recording and record again.

In a healthy flamegraph nearly all the width sits under the CLI's own outer
frames. `perf` copies a bounded region of the stack per sample, and a call
chain deeper than that region is cut at its outer end. Chains that reach the
top start at `_start` and `main`, cut chains start at a `novomodelo::` function
of the command layer, and worker threads start at `clone3`. A large share of
the width under roots such as `malloc`, `free` or a solver function means the
stacks did not unwind.

### Interactive views

```bash
perf report --no-children                          # terminal browser, sorted by self time
perf annotate --stdio -l 'HFactor::buildSimple()'  # source lines inside one function
hotspot perf.data                                  # GUI: flamegraph, caller/callee, timeline
```

All three read the `perf.data` that `flamegraph` recorded. `perf annotate`
takes the function name as `perf report` prints it. To record without
rendering a flamegraph, run `taskset -c "$PCORES" perf $PERF_RECORD -- "$BIN" …`.

### Callgrind

Callgrind counts the instructions each function executes instead of sampling,
so its numbers do not depend on clock frequency or on other load on the
machine. That suits it to comparing two versions of a function on the same
input.

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out.novomodelo \
  "$BIN" run /tmp/4ree-short --quiet
callgrind_annotate callgrind.out.novomodelo | head -40
kcachegrind callgrind.out.novomodelo
```

## 7. Heap profiling

### DHAT: allocations on the training path

The `dhat_baseline` example in `crates/novomodelo-sddp/examples/` trains a fixed
deterministic case with DHAT recording every heap allocation. Use it to check
that a change adds no allocations to the training loop:

```bash
cargo run --profile profiling -p novomodelo-sddp --example dhat_baseline --features dhat-heap
```

It writes `dhat-heap.json` to the current directory and prints the total,
peak (`t-gmax`) and end-of-run allocation counts. Open
`/usr/libexec/valgrind/dh_view.html` (installed with Valgrind) in a browser
and load the file to see the allocation sites ranked by bytes or by number of
blocks.

### heaptrack: a whole CLI run

`heaptrack` needs no feature flag and works on the CLI binary:

```bash
heaptrack --record-only "$BIN" run /tmp/4ree-short --quiet
heaptrack_print heaptrack.novomodelo.*.zst | less
heaptrack_gui heaptrack.novomodelo.*.zst
```

Without `--record-only`, `heaptrack` opens `heaptrack_gui` itself when the
recording ends and waits until you close it. The report gives peak heap use,
the call sites that allocate, temporary allocations (freed right after they
are made) and memory still allocated at exit, which `heaptrack` lists as
leaked; global state alive at exit shows up there too.

## 8. Memory errors

Rust code cannot read freed or uninitialised memory outside `unsafe`, so
Memcheck is most useful at the boundary with the vendored C/C++ solvers:

```bash
valgrind --leak-check=full --errors-for-leak-kinds=definite --error-exitcode=1 \
  "$BIN" run /tmp/4ree-short --quiet
```

The summary at the end should read `ERROR SUMMARY: 0 errors` and
`definitely lost: 0 bytes`, and the command exits with status 0.

The CLI's worker threads, from the `rayon` global thread pool, are still
running when the process exits, so Memcheck can report their per-thread data
as `possibly lost`. These blocks are allocated by `pthread_create` or by
`std::thread` code, with `rayon` further down the stack, and which of them
appear varies from run to run. `--errors-for-leak-kinds=definite` keeps them
out of the error count; they are still printed. Look into any
`possibly lost` record allocated elsewhere. Blocks listed as `still reachable`
are global state alive at exit, not leaks.

## 9. Troubleshooting

| Symptom                                                                          | Cause and fix                                                                                                                                            |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `perf_event_paranoid setting is 4`                                               | The kernel setting from [section 3](#3-allow-perf-to-sample-your-own-processes) is missing.                                                             |
| `WARNING: perf not found for kernel …`                                           | No `linux-tools` package for the running kernel; see [section 2](#2-install-the-tools).                                                                 |
| Flamegraph is mostly `malloc`, `free`, `operator new` or `[unknown]` at the root | Broken stacks. Run the [stack check](#checking-that-the-stacks-unwound): rebuild with `scripts/build-profiling.sh`, record with `PERF_RECORD`, and record again if one recording fails. |
| Flamegraph holds only a few frames, such as `taskset` or `perf` start-up         | It drew a sample from an efficiency core and dropped the rest ([section 5](#5-shell-setup-and-workloads)). Put `taskset -c "$PCORES"` in front of `flamegraph`. |
| `perf record` prints `Check IO/CPU overload!`                                    | Samples were lost while writing `perf.data`. Lower `-F` in `PERF_RECORD`.                                                                             |
| `heaptrack` does not return after the program ends                              | It opened `heaptrack_gui`; close it, or pass `--record-only`.                                                                                         |
| `cargo` rebuilds everything unexpectedly                                         | `cargo` ran outside the repository, so `rustup` used your default toolchain instead of the pinned one. Run `cargo` from the repository root.            |
| Solver frames show function names but no source lines                            | The binary was not built with the `profiling` profile; the other profiles compile the vendored solvers without line tables.                            |
| Disk filling up                                                                  | See below.                                                                                                                                               |

Profiling uses disk in three places:

- `target/perf/` is a full separate build.
- `perf.data` grows with run length and sampling rate; lower `-F` in
  `PERF_RECORD` for long runs.
- `perf record` caches a copy of every binary it profiles in `~/.debug`, and
  each rebuild adds another copy. Check it with `du -sh ~/.debug`. Deleting the
  directory is safe; `perf report` on an old `perf.data` then cannot resolve
  symbols of binaries that have since been rebuilt.

## 10. Cleaning up

```bash
rm -f perf.data perf.data.old flamegraph.svg callgrind.out.* heaptrack.*.zst dhat-heap.json
rm -rf target/perf /tmp/4ree-short
```
