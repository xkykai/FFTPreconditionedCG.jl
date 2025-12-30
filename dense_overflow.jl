using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Models: buoyancy_field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Grids: with_number_type, xnodes, ynodes, znodes
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

const Nx = 1536
const Ny = 640
const Nz = 128
# const Ny = 64
# const Nz = 12
const Δx = Δy = 500
const Δz = 30
# const Δx = 500
# const Δy = 5000
# const Δz = 300
const Lx = Nx * Δx
const Ly = Ny * Δy
const Lz = Nz * Δz
# const Lz = Nz * Δz

const x₁ = 100e3
const x₀ = x₁ - Lx
const y₁ = 50e3
const y₀ = y₁ - Ly
const z₁ = 0
const z₀ = z₁ - Lz
const hₑ = 600 # depth of embayment
const Wₑ = 100e3 # width of embayment
const xₑ = (-Wₑ / 2, Wₑ / 2) # x extent of embayment
const b₀ = 0
const Δb₀ = 0.019
const U₀ = 2.4
const N = 2.3e-3
const f₀ = 1e-4
const tanα = 0.01
const h₀ = 300
const Lρ = sqrt(Δb₀ * h₀) / f₀
const Cd = 2e-3

function bathymetry(x, y)
    if y <= 0
        return -hₑ + y * tanα
    elseif x >= xₑ[1] && x <= xₑ[2]
        return -hₑ
    else
        return 0
    end
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
                        topology = (Bounded, Bounded, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

#%%
# fig = Figure()
# ax = Axis(fig[1, 1]; title = "Bathymetry", xlabel = "x (m)", ylabel = "y (m)")myq
# hm = heatmap!(ax, xnodes(grid, Center()), ynodes(grid, Center()), interior(grid.immersed_boundary.bottom_height, :, :, 1), colormap=:plasma)
# Colorbar(fig[1, 2], hm; label = "Depth (m)")
# display(fig)
#%%
@inline h_inflow(xwall) = h₀ * exp(-xwall / Lρ)
@inline compute_zstar(z, xwall) = (z - h_inflow(xwall) + hₑ) / h_inflow(xwall)

@inline function F(zstar)
    Riₘ = 1 // 3

    if zstar >= Riₘ / (2 - Riₘ)
        return 1
    elseif zstar > -Riₘ / (2 + Riₘ) && zstar < Riₘ / (2 - Riₘ)
        return 1 / Riₘ * zstar / (zstar + 1) + 1/2
    else
        return 0
    end
end

zs = -600:0
xwalls = 0:100:100e3

zstars = [compute_zstar.(z, xwall) for xwall in xwalls, z in zs]
Fs = F.(zstars)

#%%
# fig = Figure()
# ax = Axis(fig[1, 1]; title = "Inflow profile function F(z*)", xlabel = "x (m)", ylabel = "z (m)")
# hm = heatmap!(ax, xwalls, zs, Fs; colormap=:plasma)
# Colorbar(fig[1, 2], hm; label = "F(z*)")
# display(fig)
#%%
@inline u_quadratic_drag(x, y, t, u, v) = - Cd * u * sqrt(u^2 + v^2)
@inline v_quadratic_drag(x, y, t, u, v) = - Cd * v * sqrt(u^2 + v^2)

u_quadratic_drag_bc = FluxBoundaryCondition(u_quadratic_drag, field_dependencies=(:u, :v))
v_quadratic_drag_bc = FluxBoundaryCondition(v_quadratic_drag, field_dependencies=(:u, :v))

@inline function v_inflow_profile(x, y, z, t)
    xwall = x + Wₑ / 2
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return -U₀ * exp(-xwall / Lρ) * (1 - Fstar)
end

@inline function b_inflow_profile(x, y, z, t)
    xwall = x + Wₑ / 2
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return min(b₀ - Δb₀ * (1 - Fstar), b₀ + N^2 * z)
end

@inline function c_inflow_profile(x, y, z, t)
    xwall = x + Wₑ / 2
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return ifelse(Fstar != 0, 1, 0)
end

inflow_mask = GaussianMask{:y}(center = y₁, width = abs(y₁)/5)

Δt = Δy / abs(U₀) / 5
damping_rate = 1 / (Δt * 20)

v_inflow_forcing = Relaxation(rate = damping_rate, 
                              target = v_inflow_profile, 
                              mask = inflow_mask)

b_inflow_forcing = Relaxation(rate = damping_rate, 
                              target = b_inflow_profile, 
                              mask = inflow_mask)

c_inflow_forcing = Relaxation(rate = damping_rate, 
                              target = c_inflow_profile, 
                              mask = inflow_mask)

@inline b_background(x, y, z, t) = N^2 * z

left_sponge_mask = GaussianMask{:x}(center = x₀, width = 10e3)
zero_left_sponge = Relaxation(rate = damping_rate, mask = left_sponge_mask)
b_left_sponge = Relaxation(rate = damping_rate, target = b_background, mask = left_sponge_mask)

u_forcing = zero_left_sponge
v_forcing = (v_inflow_forcing, zero_left_sponge)
w_forcing = zero_left_sponge
b_forcing = (b_inflow_forcing, b_left_sponge)
c_forcing = (c_inflow_forcing, zero_left_sponge)

forcing = (u = u_forcing, v = v_forcing, w = w_forcing, b = b_forcing, c = c_forcing)
#%%
@inline function immersed_u_boundary_condition(x, y, z, t, u, v)
    within_embayment = z >= h₀
    return ifelse(within_embayment, 0, u_quadratic_drag(x, y, t, u, v))
end

@inline function immersed_v_boundary_condition(x, y, z, t, u, v)
    within_embayment = z >= h₀
    return ifelse(within_embayment, 0, v_quadratic_drag(x, y, t, u, v))
end

immersed_u_bc = FluxBoundaryCondition(immersed_u_boundary_condition, field_dependencies=(:u, :v))
immersed_v_bc = FluxBoundaryCondition(immersed_v_boundary_condition, field_dependencies=(:u, :v))

no_slip_bc = ValueBoundaryCondition(0)
no_flux_bc = FluxBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(immersed=immersed_u_bc, top=no_slip_bc, bottom=u_quadratic_drag_bc, north=no_slip_bc, south=no_slip_bc)
v_bcs = FieldBoundaryConditions(immersed=immersed_v_bc, top=no_slip_bc, bottom=v_quadratic_drag_bc, east=no_slip_bc, west=no_slip_bc)
w_bcs = FieldBoundaryConditions(immersed=no_slip_bc, north=no_slip_bc, south=no_slip_bc, east=no_slip_bc, west=no_slip_bc)
b_bcs = FieldBoundaryConditions(no_flux_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, b = b_bcs)
#%%
# reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
# preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
# pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100; preconditioner)
# pressure_solver_str = "CG"
pressure_solver = nothing
pressure_solver_str = "FFT"
#%%
filename = "dense_overflow_Nx_$(Nx)_Ny_$(Ny)_Nz_$(Nz)_$(pressure_solver_str)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)

model = NonhydrostaticModel(; grid, pressure_solver,
                              advection = WENO(order=9),
                              tracers = (:b, :c),
                              buoyancy = BuoyancyTracer(),
                              boundary_conditions,
                              forcing)
#%%
bᵢ(x, y, z) = b_background(x, y, z, nothing) + rand() * 1e-5

set!(model, b=bᵢ)

stop_time = 1day

simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(10))

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