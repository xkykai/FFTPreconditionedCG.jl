using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Models: buoyancy_field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Grids: with_number_type, xnodes, ynodes, znodes, xnode, znode
using Oceananigans.Utils: launch!
using Oceananigans.Units
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA
using CairoMakie
using NaNStatistics

arch = GPU()
# arch = CPU()

const Nx = 1536
const Ny = 512
const Nz = 128
# const Nx = 128
# const Ny = 64
# const Nz = 12
const Δx = 500
const Δy = 500
const Δz = 30
# const Δx = 500
# const Δy = 5000
# const Δz = 300
const Lx = Nx * Δx
const Ly = Ny * Δy
const Lz = Nz * Δz

const x₁ = 100e3
const x₀ = x₁ - Lx
const y₁ = 0
const y₀ = y₁ - Ly
const z₁ = 0
const z₀ = z₁ - Lz
const hₑ = 600 # depth of embayment
const Wₑ = 100e3 # width of embayment
const xₑ = (-Wₑ / 2, Wₑ / 2) # x extent of embayment
const b₀ = 0
const Δb₀ = 0.019
const N = 2.3e-3
const f₀ = 1e-4
const tanα = 0.01
const Cd = 2e-3

function bathymetry(x, y)
    return -hₑ + y * tanα
end

#####
##### Model setup
#####

grid = RectilinearGrid(arch, Float64,
                        size = (Nx, Ny, Nz), 
                        halo = (6, 6, 6),
                        x = (x₀, x₁),
                        y = (y₀, y₁),
                        z = (z₀, z₁),
                        topology = (Periodic, Bounded, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

#%%
# fig = Figure()
# ax = Axis(fig[1, 1]; title = "Bathymetry", xlabel = "x (m)", ylabel = "y (m)")
# hm = heatmap!(ax, xnodes(grid, Center()), ynodes(grid, Center()), Array(interior(grid.immersed_boundary.bottom_height, :, :, 1)), colormap=:plasma)
# Colorbar(fig[1, 2], hm; label = "Depth (m)")
# display(fig)
#%%
@inline u_quadratic_drag(x, y, z, t, u, v) = - Cd * u * sqrt(u^2 + v^2)
@inline v_quadratic_drag(x, y, z, t, u, v) = - Cd * v * sqrt(u^2 + v^2)

@inline u_quadratic_bottom_drag(x, y, t, u, v) = u_quadratic_drag(x, y, nothing, t, u, v)
@inline v_quadratic_bottom_drag(x, y, t, u, v) = v_quadratic_drag(x, y, nothing, t, u, v)

u_quadratic_bottom_drag_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, field_dependencies=(:u, :v))
v_quadratic_bottom_drag_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, field_dependencies=(:u, :v))

immersed_u_bc = FluxBoundaryCondition(u_quadratic_drag, field_dependencies=(:u, :v))
immersed_v_bc = FluxBoundaryCondition(v_quadratic_drag, field_dependencies=(:u, :v))

advective_Δt = sqrt(Δz / Δb₀)
τ = advective_Δt * 10

@inline function b_inflow_profile(i, k, grid, clock, model_fields)
    @inbounds x = xnode(i, grid, Center())
    @inbounds z = znode(k, grid, Center())
    within_zone = (x >= -Wₑ / 2) & (x <= Wₑ / 2) & (z >= -hₑ)
    @inbounds b = model_fields.b[i, Ny, k]
    Δb = (b₀ - Δb₀) - b
    return ifelse(within_zone, -Δb * Δy / τ, 0)
end

@inline function c_inflow_profile(i, k, grid, clock, model_fields)
    @inbounds x = xnode(i, grid, Center())
    @inbounds z = znode(k, grid, Center())
    within_zone = (x >= -Wₑ / 2) & (x <= Wₑ / 2) & (z >= -hₑ)
    @inbounds c = model_fields.c[i, Ny, k]
    Δc = 1 - c
    return ifelse(within_zone, -Δc * Δy / τ, 0)
end

b_inflow_bc = FluxBoundaryCondition(b_inflow_profile, discrete_form=true)
c_inflow_bc = FluxBoundaryCondition(c_inflow_profile, discrete_form=true)

no_slip_bc = ValueBoundaryCondition(0)
no_flux_bc = FluxBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(immersed=immersed_u_bc, bottom=u_quadratic_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(immersed=immersed_v_bc, bottom=v_quadratic_bottom_drag_bc)
b_bcs = FieldBoundaryConditions(north=b_inflow_bc)
c_bcs = FieldBoundaryConditions(north=c_inflow_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs, c = c_bcs)
#%%
reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100; preconditioner)
pressure_solver_str = "CG"
# pressure_solver = nothing
# pressure_solver_str = "FFT"
#%%
coriolis = FPlane(f₀)
#%%
filename = "dense_overflow_noinlet_Nx_$(Nx)_Ny_$(Ny)_Nz_$(Nz)_$(pressure_solver_str)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)

model = NonhydrostaticModel(; grid, pressure_solver,
                              advection = WENO(order=9),
                              tracers = (:b, :c),
                              coriolis,
                              buoyancy = BuoyancyTracer(),
                              boundary_conditions)
#%%
@inline b_background(x, y, z, t) = N^2 * z
bᵢ(x, y, z) = b_background(x, y, z, nothing) + rand() * 1e-5 * abs(N^2 * Δz)

set!(model, b=bᵢ)

stop_time = 5days

Δt = advective_Δt / 5

simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
b, c = model.tracers.b, model.tracers.c

d = CenterField(grid)

@kernel function _divergence!(target_field, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_flow_divergence!(target_field, model)
    grid = model.grid
    u, v, w = model.velocities
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
    return nothing
end

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("i: %d, t: %s, wall t: %s, Δt: %s, Poisson: %d",
                    iteration(sim), prettytime(sim), prettytime(elapsed), prettytime(sim.Δt), pressure_iters)

    compute_flow_divergence!(d, sim.model)

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max b: %6.3e, max c: %6.3e, max d: %6.3e, max p: %6.3e, mean p: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.b),
                    maximum(abs, sim.model.tracers.c),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    mean(sim.model.pressures.pNHS),
    )

    wall_clock[] = time_ns()
    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

simulation.output_writers[:jld2] = JLD2Writer(model, (; b, c);
                                              filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                              schedule = TimeInterval(6hours),
                                              with_halos = true,
                                              overwrite_existing = true)

run!(simulation)
#%%