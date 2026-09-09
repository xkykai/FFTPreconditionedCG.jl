using CUDA
using MPI
using Oceananigans
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

communicator(arch::Distributed) = arch.communicator
communicator(arch) = nothing

solver_iterations(solver::ConjugateGradientPoissonSolver) = solver.conjugate_gradient_solver.iteration
solver_iterations(solver) = 0

gpu_state() = let dev = CUDA.NVML.Device(CUDA.uuid(CUDA.device()))
    (sm_clock = CUDA.NVML.clock_info(dev).sm,
     temperature = CUDA.NVML.temperature(dev),
     power = CUDA.NVML.power_usage(dev))
end

"""
    benchmark_time_steps!(model, Δt, nsteps; warmup)

Time `nsteps` calls to `time_step!`, returning per-step `wall` time in seconds, the pressure
solver `iterations` per step, and the `elapsed` time of the whole loop measured between rank
barriers.

Each step is synchronized before the host timestamp is taken, so `wall` is the time to
complete the step rather than the time to queue it. `elapsed / nsteps` is the mean step time,
which memory pool stalls inflate well above `median(wall)`.
"""
function benchmark_time_steps!(model, Δt, nsteps; warmup = nsteps)
    comm = communicator(model.architecture)
    barrier() = isnothing(comm) || MPI.Barrier(comm)

    for _ in 1:warmup
        time_step!(model, Δt)
    end

    wall = zeros(nsteps)
    iterations = zeros(Int, nsteps)

    CUDA.synchronize()
    barrier()
    initial_state = gpu_state()
    t₀ = time_ns()

    for n in 1:nsteps
        tₙ = time_ns()
        time_step!(model, Δt)
        CUDA.synchronize()
        wall[n] = (time_ns() - tₙ) * 1e-9
        iterations[n] = solver_iterations(model.pressure_solver)
    end

    barrier()
    elapsed = (time_ns() - t₀) * 1e-9

    return (; wall, iterations, elapsed, initial_state, final_state = gpu_state())
end
