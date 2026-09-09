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
using ArgParse
using Random

include("benchmark_utils.jl")

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

const N = 480
const Ra = 1e6
const ν = κ = 1 / sqrt(Ra)

#####
##### Model setup
#####

@inline function local_roughness_bottom(η, η₀, h)
    if η > η₀ - h && η <= η₀
        return η + h - η₀
    elseif η > η₀ && η <= η₀ + h
        return -η + h + η₀
    else
        return 0
    end
end

@inline function local_roughness_top(η, η₀, h)
    if η > η₀ - h && η <= η₀
        return -η - h + η₀
    elseif η > η₀ && η <= η₀ + h
        return η - h - η₀
    else
        return 0
    end
end

function setup_grid()
    Lx = ngpus
    Ly = Lz = 1
    Nx = N * ngpus
    Ny = Nz = N

    grid = RectilinearGrid(arch, Float64,
                           size = (Nx, Ny, Nz), 
                           halo = (6, 6, 6),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           topology = (Bounded, Bounded, Bounded))

    slope(x, y) = 0.35

    Nr_x = 8 * ngpus # number of roughness elements in x-direction
    Nr_y = 8 # number of roughness elements in y-direction

    hx = Lx / Nr_x / 2
    hy = Ly / Nr_y / 2
    x₀s = hx:2hx:Lx-hx
    y₀s = hy:2hy:Ly-hy

    # Woven pattern: ridges in x and y directions create pyramids
    @inline function roughness_bottom(x, y, z)
        # Ridges running in y-direction (triangular profile in x)
        z_rough_x = sum([local_roughness_bottom(x, x₀, hx) for x₀ in x₀s])
        # Ridges running in x-direction (triangular profile in y)
        z_rough_y = sum([local_roughness_bottom(y, y₀, hy) for y₀ in y₀s])
        # Take minimum to create pyramidal peaks
        z_rough = min(z_rough_x, z_rough_y)
        return z <= z_rough
    end

    @inline function roughness_top(x, y, z)
        z_rough_x = sum([local_roughness_top(x, x₀, hx) for x₀ in x₀s])
        z_rough_y = sum([local_roughness_top(y, y₀, hy) for y₀ in y₀s])
        # Take maximum for inverted pyramids
        z_rough = max(z_rough_x, z_rough_y)
        return z >= z_rough + Lz
    end

    @inline mask(x, y, z) = roughness_bottom(x, y, z) | roughness_top(x, y, z)

    grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(mask))
    return grid
end

function initial_conditions!(model)
    Random.seed!(1234 + local_rank)
    bᵢ(x, y, z) = rand() * 1e-2 - z + 0.5

    set!(model, b=bᵢ)
end

function setup_model(grid, pressure_solver)
    closure = ScalarDiffusivity(ν=ν, κ=κ)

    @inline function rayleigh_benard_buoyancy(x, y, z, t)
        above_centerline = z > 1 / 2
        return ifelse(above_centerline, -1/2, 1/2)
    end

    no_slip_bc = ValueBoundaryCondition(0)

    u_bcs = FieldBoundaryConditions(no_slip_bc)
    v_bcs = FieldBoundaryConditions(no_slip_bc)
    w_bcs = FieldBoundaryConditions(no_slip_bc)
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(-1/2), bottom=ValueBoundaryCondition(1/2),
                                    immersed=ValueBoundaryCondition(rayleigh_benard_buoyancy))

    model = NonhydrostaticModel(grid; pressure_solver,
                                  advection = WENO(order=9),
                                  closure = closure,
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer(),
                                  boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))

    initial_conditions!(model)
    return model
end

function build_solver(grid, precond_name)
    precond_name == "FFT" && return nothing

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

    reltol = 100 * eps(grid) * volume^2
    abstol = 100 * eps(grid)

    return ConjugateGradientPoissonSolver(grid, maxiter=10000; reltol, abstol, preconditioner)
end

Δt = min(1 / N, (1/N^2) / max(ν, κ)) / 3

warmup_nsteps = 50
nsteps = 50
nrepeats = 2

preconditioners = ["FFT", "no", "FFT64", "FFT32", "MITgcm"]

local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
OUTPUT_DIR = "./reports/weakscaling_H100_timed_nogc/benchmark_$(ngpus)gpu"

mkpath(OUTPUT_DIR)
FILE_PATH = joinpath(OUTPUT_DIR, "rank_$(local_rank)_timed.jld2")
isfile(FILE_PATH) && rm(FILE_PATH)

for repeat in 1:nrepeats
    # Alternating order cancels drift in GPU throughput over the sweep
    order = isodd(repeat) ? preconditioners : reverse(preconditioners)

    for precond_name in order
        @info "Benchmarking $precond_name on rank $local_rank, repeat $repeat"

        grid = setup_grid()
        model = setup_model(grid, build_solver(grid, precond_name))

        results = benchmark_time_steps!(model, Δt, nsteps; warmup=warmup_nsteps)

        jldopen(FILE_PATH, "a") do file
            file["$(precond_name)/$(repeat)"] = results
        end

        grid = nothing
        model = nothing
        GC.gc()
        CUDA.reclaim()
    end
end
