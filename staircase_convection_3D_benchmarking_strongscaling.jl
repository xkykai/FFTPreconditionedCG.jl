using MPI
using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner
using Oceananigans.Grids: with_number_type
using Oceananigans.DistributedComputations
using Statistics
using CUDA
using NVTX
using ArgParse
# using CairoMakie

MPI.Init()

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--ngpus"
        help = "Number of GPUs to use"
        arg_type = Int
        default = 1
    end
    return parse_args(s)
end

args = parse_commandline()
ngpus = args["ngpus"]

if MPI.Comm_size(MPI.COMM_WORLD) == 1
    arch = GPU()
else
    arch = Distributed(GPU(); partition = Partition(x = DistributedComputations.Equal()), synchronized_communication=false)
end

N = 480

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid()
    grid = RectilinearGrid(arch, Float64,
                           size = (N, N, N), 
                           halo = (6, 6, 6),
                           x = (0, 1),
                           y = (0, 1),
                           z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))

    slope(x, y) = 0.35

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(order=9),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

function setup_simulation(model, Δt, stop_iteration)
    return Simulation(model, Δt=Δt, stop_iteration=stop_iteration)
end

Δt = 2e-2 * 64 / 2 / N
nsteps = 100

@info "Benchmarking FFT solver"
grid = setup_grid()
pressure_solver = nothing
model = setup_model(grid, pressure_solver)

for step in 1:3
    time_step!(model, Δt)
end

for step in 1:nsteps
    NVTX.@range "FFT timestep" begin
        time_step!(model, Δt)
    end
end

preconditioners = ["no", "FFT64", "FFT32", "MITgcm"]

for precond_name in preconditioners
    @info "Benchmarking $precond_name preconditioner"
    global grid = nothing
    global model = nothing
    global pressure_solver = nothing
    global preconditioner = nothing
    GC.gc()
    CUDA.reclaim()

    grid = setup_grid()
    if precond_name == "no"
        preconditioner = nothing
    elseif precond_name == "FFT64"
        preconditioner = nonhydrostatic_pressure_solver(arch, grid.underlying_grid, nothing)
    elseif precond_name == "FFT32"
        reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
        preconditioner = nonhydrostatic_pressure_solver(arch, reduced_precision_grid, nothing)
    elseif precond_name == "MITgcm"
        preconditioner = DiagonallyDominantPreconditioner()
    end

    volume = grid.Δxᶜᵃᵃ * grid.Δyᵃᶜᵃ * grid.z.Δᵃᵃᶜ
    cg_iters = zeros(nsteps)

    reltol = 100 * eps(grid) * volume^2
    abstol = 100 * eps(grid)

    pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10000; reltol, abstol, preconditioner)

    model = setup_model(grid, pressure_solver)

    for step in 1:3
        time_step!(model, Δt)
    end

    for step in 1:nsteps
        NVTX.@range "$(precond_name) preconditioner" begin
            time_step!(model, Δt)
        end

        if ngpus == 1 || model.architecture.local_rank == 0
            cg_iters[step] = model.pressure_solver.conjugate_gradient_solver.iteration        
        end
    end

    if ngpus == 1 || model.architecture.local_rank == 0
        mkpath("./reports/strongscaling_H100")
        jldopen("./reports/strongscaling_H100/cg_iters.jld2", "a") do file
            file["$(precond_name)"] = cg_iters
        end
    end
end