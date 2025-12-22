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
using BenchmarkTools

arch = GPU()

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

function setup_grid(N)
    Lx = Ly = Lz = 1

    grid = RectilinearGrid(arch, Float64,
                           size = (N, N, N), 
                           halo = (6, 6, 6),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           topology = (Bounded, Bounded, Bounded))

    slope(x, y) = 0.35

    Nr = 8 # number of roughness elements
    hx = Lx / Nr / 2
    hy = Ly / Nr / 2
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

    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(order=9),
                                  closure = closure,
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer(),
                                  boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))

    initial_conditions!(model)
    return model
end

Ns = [32, 64, 96, 128, 192, 256, 384, 512]
Δts = [min(1 / N, (1/N^2) / max(ν, κ)) / 3 for N in Ns]

mkpath("./reports/")
filename = "single_H100_timed.jld2"
FILE_PATH = joinpath("./reports/", filename)

warmup_nsteps = 50
nsteps = 50

for (N, Δt) in zip(Ns, Δts)
    @info "Benchmarking FFT solver for N=$N"
    grid = setup_grid(N)
    pressure_solver = nothing
    model = setup_model(grid, pressure_solver)
    times_FFT = []

    for step in 1:warmup_nsteps
        time_step!(model, Δt)
    end

    for step in 1:nsteps
        GC.gc()
        CUDA.reclaim()
        t = @timed time_step!(model, Δt)
        push!(times_FFT, t)
    end

    jldopen(FILE_PATH, "a") do file
        file["$(N)/times/FFTstep"] = times_FFT
    end
    
    preconditioners = ["no", "FFT64", "FFT32", "MITgcm"]

    for precond_name in preconditioners
        @info "Benchmarking $precond_name preconditioner for N=$N"
        grid = nothing
        model = nothing
        pressure_solver = nothing
        preconditioner = nothing
        GC.gc()
        CUDA.reclaim()

        grid = setup_grid(N)
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

        pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10000; reltol, abstol, preconditioner)

        model = setup_model(grid, pressure_solver)

        for step in 1:warmup_nsteps
            time_step!(model, Δt)
        end

        cg_iters = Int[]
        times = []

        for step in 1:nsteps
            GC.gc()
            CUDA.reclaim()
            t = @timed time_step!(model, Δt)
            push!(times, t)
            push!(cg_iters, model.pressure_solver.conjugate_gradient_solver.iteration)
        end

        jldopen(FILE_PATH, "a") do file
            file["$(N)/times/$(precond_name)"] = times
            file["$(N)/cg_iters/$(precond_name)"] = cg_iters
        end
    end
end