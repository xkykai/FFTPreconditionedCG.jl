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
# using GLMakie
using NaNStatistics

arch = GPU()
# arch = CPU()

const Nx = 1536
const Ny = 640
const Nz = 128
# const Nx = 128
# const Ny = 64
# const Nz = 128
const Δx = 500
const Δy = 500
const Δz = 30
# const Δx = 5000
# const Δy = 5000
# const Δz = 30
const Lx = Nx * Δx
const Ly = Ny * Δy
const Lz = Nz * Δz

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
# ax = Axis(fig[1, 1]; title = "Bathymetry", xlabel = "x (m)", ylabel = "y (m)")
# hm = heatmap!(ax, grid.immersed_boundary.bottom_height, colormap=:plasma)
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

@inline compute_xwall(x) = x + Wₑ / 2
#%%
# zs = -600:0
# xwalls = 0:100:100e3

# zstars = [compute_zstar.(z, xwall) for xwall in xwalls, z in zs]
# Fs = F.(zstars)

# fig = Figure()
# ax = Axis(fig[1, 1]; title = "Inflow profile function F(z*)", xlabel = "x (m)", ylabel = "z (m)")
# hm = heatmap!(ax, xwalls, zs, Fs; colormap=:plasma)
# Colorbar(fig[1, 2], hm; label = "F(z*)")
# display(fig)
#%%
@inline function v_inflow_profile(x, z, t)
    xwall = compute_xwall(x)
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return -U₀ * exp(-xwall / Lρ) * (1 - Fstar)
end

@inline function b_inflow_profile(x, z, t)
    xwall = compute_xwall(x)
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return min(b₀ - Δb₀ * (1 - Fstar), b₀ + N^2 * z)
end

@inline function c_inflow_profile(x, z, t)
    xwall = compute_xwall(x)
    zstar = compute_zstar(z, xwall)
    Fstar = F(zstar)
    return ifelse(Fstar != 0, 1, 0)
end

@inline function bathymetry_aware_v_inlet_profile(x, y, z)
    h = bathymetry(x, y)
    ifelse(z < h, zero(x), v_inflow_profile(x, z, nothing))
end

@inline bathymetry_aware_v_inlet_profile(x, z) = bathymetry_aware_v_inlet_profile(x, y₁, z)

inlet_field = Field{Center, Nothing, Center}(grid)
set!(inlet_field, bathymetry_aware_v_inlet_profile)

inlet_mass_flux_total = Field(Integral(inlet_field))

Array(interior(inlet_mass_flux_total, 1, 1, 1))[1]
#%%
# fig = Figure()
# ax = Axis(fig[1, 1]; title = "Inlet v profile", xlabel = "x (m)", ylabel = "z (m)")
# hm = heatmap!(ax, inlet_field)
# Colorbar(fig[1, 2], hm; label = "Inlet v (m/s)")
# display(fig)
#%%
@inline function bathymetry_aware_mask(x, y, z)
    h = bathymetry(x, y)
    return ifelse(z < h, zero(x), one(x))
end

@inline west_wall_mask(y, z) = bathymetry_aware_mask(x₀, y, z)

west_wall_field = Field{Nothing, Face, Center}(grid)
set!(west_wall_field, west_wall_mask)

outlet_area = Field(Integral(west_wall_field))

const U₀_out = abs(Array(interior(inlet_mass_flux_total, 1, 1, 1))[1] / Array(interior(outlet_area, 1, 1, 1))[1])

u_outlet(y, z, t) = -U₀_out
#%%
# fig = Figure()
# ax = Axis(fig[1, 1]; title = "West outlet area field", xlabel = "y (m)", ylabel = "z (m)")
# hm = heatmap!(ax, west_outlet_field)
# Colorbar(fig[1, 2], hm; label = "Area (m²)")
# display(fig)
#%%
v_inlet_bc = OpenBoundaryCondition(v_inflow_profile; scheme=PerturbationAdvection())
b_inlet_bc = ValueBoundaryCondition(b_inflow_profile)
c_inlet_bc = ValueBoundaryCondition(c_inflow_profile)

u_west_outlet_bc = OpenBoundaryCondition(u_outlet; scheme=PerturbationAdvection())

@inline u_quadratic_drag(x, y, z, t, u, v) = - Cd * u * sqrt(u^2 + v^2)
@inline v_quadratic_drag(x, y, z, t, u, v) = - Cd * v * sqrt(u^2 + v^2)

@inline u_quadratic_bottom_drag(x, y, t, u, v) = u_quadratic_drag(x, y, nothing, t, u, v)
@inline v_quadratic_bottom_drag(x, y, t, u, v) = v_quadratic_drag(x, y, nothing, t, u, v)

u_quadratic_bottom_drag_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, field_dependencies=(:u, :v))
v_quadratic_bottom_drag_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, field_dependencies=(:u, :v))

@inline function immersed_u_boundary_condition(x, y, z, t, u, v)
    within_embayment = y >= 0
    return ifelse(within_embayment, zero(u), u_quadratic_drag(x, y, z, t, u, v))
end

@inline function immersed_v_boundary_condition(x, y, z, t, u, v)
    within_embayment = y >= 0
    return ifelse(within_embayment, zero(v), v_quadratic_drag(x, y, z, t, u, v))
end

immersed_u_bc = FluxBoundaryCondition(immersed_u_boundary_condition, field_dependencies=(:u, :v))
immersed_v_bc = FluxBoundaryCondition(immersed_v_boundary_condition, field_dependencies=(:u, :v))

no_slip_bc = ValueBoundaryCondition(0)
no_flux_bc = FluxBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(immersed=immersed_u_bc, bottom=u_quadratic_bottom_drag_bc, north=no_slip_bc, west=u_west_outlet_bc)
v_bcs = FieldBoundaryConditions(immersed=immersed_v_bc, bottom=v_quadratic_bottom_drag_bc, north=v_inlet_bc)
w_bcs = FieldBoundaryConditions(north=no_slip_bc)
b_bcs = FieldBoundaryConditions(north=b_inlet_bc)
c_bcs = FieldBoundaryConditions(north=c_inlet_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, b = b_bcs, c = c_bcs)
#%%
pressure_solver = ConjugateGradientPoissonSolver(grid)
pressure_solver_str = "CG"
# pressure_solver = nothing
# pressure_solver_str = "FFT"
#%%
coriolis = FPlane(f₀)
#%%
filename = "dense_overflow_Nx_$(Nx)_Ny_$(Ny)_Nz_$(Nz)_$(pressure_solver_str)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)

model = NonhydrostaticModel(; grid, pressure_solver,
                              advection = WENO(order=9),
                              tracers = (:b, :c),
                              coriolis,
                              buoyancy = BuoyancyTracer(),
                              boundary_conditions)
#%%
@inline b_background(x, y, z, t) = b₀ + N^2 * z
bᵢ(x, y, z) = b_background(x, y, z, nothing) + rand() * 1e-5 * abs(N^2 * Lz)

set!(model, b=bᵢ)

stop_time = 1day
Δt = Δy / abs(U₀) / 10
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, v, w, b, c);
                                              filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                              schedule = TimeInterval(6hours),
                                              with_halos = true,
                                              overwrite_existing = true)

run!(simulation)
# #%%
# u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u")
# v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "v")
# w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
# b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b")
# c_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "c")
# #%%
# times = u_data.times
# Nt = length(times)

# xC = xnodes(u_data.grid, Center())
# yC = ynodes(u_data.grid, Center())
# zC = znodes(u_data.grid, Center())
# xF = xnodes(u_data.grid, Face())
# yF = ynodes(u_data.grid, Face())
# zF = znodes(u_data.grid, Face())

# for i in 1:Nt
#     mask_immersed_field!(u_data[i], NaN)
#     mask_immersed_field!(v_data[i], NaN)
#     mask_immersed_field!(w_data[i], NaN)
#     mask_immersed_field!(b_data[i], NaN)
#     mask_immersed_field!(c_data[i], NaN)
# end
# #%%
# fig = Figure(size=(1000, 1000), fontsize=20)
# ax = Axis(fig[1, 1]; title = "c", xlabel = "x (m)", ylabel = "y (m)")

# slider = Slider(fig[2, :]; range = 1:length(zC), startvalue = length(zC))

# n = slider.value

# cₙ = @lift interior(c_data[Nt], :, :, $n)

# clim = @lift (-nanmaximum(abs.($cₙ)), nanmaximum(abs.($cₙ)))

# label_str = @lift "z = $(round(zC[$n]; digits=1)) m"
# Label(fig[0, :], label_str, tellwidth=false)

# hm = heatmap!(ax, xC, yC, cₙ, colorrange = clim, colormap=:balance)
# Colorbar(fig[1, 2], hm; label = "c")
# display(fig)
# #%%
# fig = Figure(size=(1000, 1000), fontsize=20)
# ax = Axis(fig[1, 1]; title = "v", xlabel = "x (m)", ylabel = "z (m)")

# slider = Slider(fig[2, :]; range = 1:length(zC), startvalue = length(zC))

# n = slider.value

# vₙ = @lift interior(v_data[Nt], :, 1:length(yF)-1, $n)

# clim = @lift (-nanmaximum(abs.($vₙ)), nanmaximum(abs.($vₙ)))

# label_str = @lift "z = $(round(zC[$n]; digits=1)) m"
# Label(fig[0, :], label_str, tellwidth=false)

# hm = heatmap!(ax, xC, yF, vₙ, colorrange = clim, colormap=:balance)
# Colorbar(fig[1, 2], hm; label = "v")
# display(fig)
# #%%