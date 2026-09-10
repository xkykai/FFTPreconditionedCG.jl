# Distributed MPI benchmarks: diagnosis and resume notes (2026-09-10)

Weak- and strong-scaling jobs 1106 and 1107 crashed at the 2-GPU stage. Two independent
defects were found and fixed; a third remains and needs the nvhpc install repaired.

**Status: distributed runs still do not work.** Single-GPU runs are unaffected.

---

## 1. Symptom

```
ERROR: LoadError: ArgumentError: a group or dataset named FFT is already present within this group
```

Both tasks logged `Benchmarking FFT on rank 0`. Every task had `MPI.Comm_rank == 0`, so MPI
never formed a multi-rank communicator. Both ranks computed `local_rank = 0`, both targeted
`benchmark_2gpu/rank_0_timed.jld2`, and the second write of `times/FFT` collided.

Only `rank_0_timed.jld2` was written that day; `rank_1_timed.jld2` was stale from 2025-12-27.
No rank 1 ever existed.

The crash was the lucky outcome. The drivers selected the architecture with

```julia
if MPI.Comm_size(MPI.COMM_WORLD) == 1
    arch = GPU()
```

so under singleton ranks every "N-GPU" run silently became a non-decomposed single-GPU run of
the full `N*ngpus` domain, all processes contending for one device. Without the filename
collision this would have produced plausible but meaningless scaling numbers.

---

## 2. Defect: MPIPreferences was a transitive dependency — FIXED

`MPI.jl` was loading `MPICH_jll` from the artifact store, not the configured HPC-X:

```
MPI_LIBRARY = MPICH
libmpi path = ~/.julia/artifacts/8bd6881e.../lib/libmpi.so   (owned by MPICH_jll)
has_cuda    = false
```

`LocalPreferences.toml` was correct and untouched, but `Preferences.jl` only applies a
`LocalPreferences.toml` block to a package reachable from the active project's *direct*
dependencies. `MPIPreferences` was transitive only, so the block was ignored and the
compile-time default (`MPICH_jll`) won.

That explains the MPICH-style errors, `--mpi=pmix` rejected as "unsupported PMI version", and
`MPI_ERR_IN_STATUS` (class 17) from `Waitall` in Oceananigans' halo exchange — that MPICH has
no CUDA awareness and Oceananigans passes device pointers straight to MPI.

Fix: `Pkg.add("MPIPreferences")`. Verified afterwards:

```
binary     = MPItrampoline_jll
abi        = OpenMPI
MPI.libmpi = .../hpcx-2.22.1/ompi/lib/libmpi.so
```

This had been silently wrong for any recent run, not just these jobs.

---

## 3. Defect: missing libpmix.so.2 symlink in HPC-X — FIXED

HPC-X ships `libpmix.so.2.2.35` but no `libpmix.so.2` symlink, so the loader resolved
`libpmix.so.2` from `/usr/local/lib` (SLURM's PMIx 4), which lacks the symbol HPC-X's PMIx 3
component needs:

```
mca_pmix_pmix3x.so: undefined symbol: pmix_value_load
```

`nm -D` confirms HPC-X's own `libpmix.so.2.2.35` defines `pmix_value_load`. This broke
`orterun` even outside SLURM, so it was never a SLURM/OpenMPI clash.

Fix applied:

```bash
ln -s libpmix.so.2.2.35 \
  ~/nvhpc/Linux_x86_64/25.7/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi/lib/libpmix.so.2
```

Verified: `mca_pmix_pmix3x.so` resolves to HPC-X's copy, and `orterun -n 2 hostname` launches
two ranks — the first working multi-rank launch.

---

## 4. Defect: HPC-X cannot open any PML component — OPEN

```
No components were able to be opened in the pml framework.
```

On both login and compute nodes: UCX is not loading. Forcing the non-UCX path
(`-mca pml ob1 -mca btl self,vader,tcp`) gets past it, then fails in `MPI_Init_thread` with
"Unknown error".

This is where work stopped.

---

## 5. Why the launcher chain broke

`MPITRAMPOLINE_LIB` points at `~/mpiwrapper-25.7/lib/libmpiwrapper.so`, and
`mpiwrapper-25.7/bin/mpiwrapperexec` is:

```sh
exec '.../hpcx-2.22.1/ompi/bin/mpiexec' "$@"
```

That path no longer exists — nvhpc 25.7 ships `ompi/bin/` containing only `env.sh`, with the
real launcher hidden at `ompi/bin/.bin/orterun`. So plain `mpiexec` fell through to
`/usr/local/bin/mpiexec`, a perl wrapper around `srun`. SLURM reports `MpiDefault = (null)`,
so `srun` launched tasks with no PMI bootstrap, producing the singleton ranks.

---

## 6. Launcher matrix

2-rank probe printing `Comm_size`:

| Launcher | Result |
|---|---|
| `mpiexec` (perl → `srun`, no plugin) | `rank=0 size=1` on both tasks |
| `srun --mpi=pmix` | unsupported PMI version / `pmix_value_load` |
| `srun --mpi=pmi2` | `size=2` under MPICH_jll; `MPI_Init_thread` fails under HPC-X |
| `orterun` (after symlink fix) | 2 ranks launch; PML then fails |

---

## 7. How to resume

Everything below is reproducible from a fresh session.

### Scratch files (may be deleted; recreate as needed)

Probes lived in `~/.claude/jobs/6affa262/tmp/`:

- `mpi_probe.jl` — prints `rank`/`size`/`libmpi`
- `mpi_caps.jl` — prints `MPI_LIBRARY`, `Get_library_version`, `has_cuda`, and greps
  `/proc/self/maps` for `libmpiwrapper`, `hpcx`, `libmpi.so.40`
- `proto_weak.jl` — the weak-scaling driver cut to `N=64`, 3 warmup + 3 steps,
  preconditioners `["FFT","FFT32"]`, output to a scratch dir. The fast end-to-end test.

Minimal `mpi_caps.jl`:

```julia
using MPI
MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("MPI_LIBRARY = ", MPI.MPI_LIBRARY)
    println("version     = ", strip(first(split(MPI.Get_library_version(), '\n'))))
    println("has_cuda    = ", MPI.has_cuda())
    println("size        = ", MPI.Comm_size(MPI.COMM_WORLD))
    maps = read("/proc/self/maps", String)
    for pat in ("libmpiwrapper", "hpcx", "libmpi.so.40")
        println("mapped $pat = ", occursin(pat, maps))
    end
end
MPI.Finalize()
```

### Running a probe

There is usually no idle allocation. Either submit a small job, or piggyback on a running one:

```bash
source $HOME/env_nvhpc_25.7.sh
srun --jobid=<RUNNING_JOBID> --overlap --cpu-bind=none --mpi=pmi2 -n 2 \
    julia +1.12.2 --startup-file=no --project mpi_caps.jl
```

Notes that cost time to rediscover:

- `--startup-file=no` is required; Revise segfaults under MPI.
- The batch scripts call `julia +1.12.2` (a juliaup channel). `~/julia-1.12.2/bin/julia`
  does **not** exist.
- Nodes are `idle~` and boot on demand; `CONFIGURING` for 20-40 min is normal, and jobs
  sometimes requeue after a failed boot.
- Each fresh Julia start compiles Oceananigans/CUDA kernels for 25-40 min. Budget for it.

### Next things to try

1. Whether `nvhpc/24.5` works as the MPIwrapper backend — it still has an intact
   `comm_libs/hpcx/bin/mpiexec`, unlike 25.7. Repoint `mpiwrapperexec` or
   `MPITRAMPOLINE_LIB` at it and rerun `mpi_caps.jl`.
2. Why UCX will not load under 25.7 HPC-X: check `ompi_info --param pml all`, whether
   `mca_pml_ucx.so` exists in `ompi/lib/openmpi/`, and whether its `libucp`/`libucs`
   dependencies resolve (`ldd`), the same failure mode as defect 3.
3. Failing both, raise with whoever maintains the nvhpc tree: the 25.7 HPC-X install is
   missing launcher binaries and a `libpmix.so.2` symlink, and its PML components do not load.

### Guard

PR #5 (branch `mpi-rank-guard`) adds:

```julia
MPI.Comm_size(MPI.COMM_WORLD) == ngpus ||
    error("launched with $(MPI.Comm_size(MPI.COMM_WORLD)) ranks but --ngpus is $ngpus")

if ngpus == 1        # was: MPI.Comm_size(MPI.COMM_WORLD) == 1
```

Keep this regardless of how the MPI problem is resolved. It turns a silent wrong-answer path
into an immediate abort, and it is what let the real failure surface downstream instead of
masquerading as a scaling result.

---

## 8. Unrelated findings

- `startup.jl` loads Revise, which segfaults under MPI (`fieldtypes_cached`,
  `_foreach_subtype!`). The batch scripts do not pass `--startup-file=no`.
- The batch scripts call `MPIPreferences.use_system_binary()` at job start, which rewrites
  `LocalPreferences.toml` every run and will overwrite the MPItrampoline setting. Remove or
  replace that call.
- `reports/single_H100/` and `reports/{weak,strong}scaling_H100/` are shared between the
  rough-RB NVTX drivers and the staircase benchmarks; running both overwrites the same
  `cg_iters.jld2`. Pre-existing.

---

## 9. Changes made

| Path | Change |
|---|---|
| `Project.toml` | added `MPIPreferences` dependency |
| `LocalPreferences.toml` | `binary = "MPItrampoline_jll"` (was `"system"`) |
| `Manifest.toml` | updated by `Pkg.add` |
| `.../hpcx-2.22.1/ompi/lib/libpmix.so.2` | new symlink |

Originals are in `backup_pre_mpifix_20260910/`; the symlink reverts with `rm`.

Keep defects 2 and 3 fixed — both are genuine and independent of the remaining problem.
