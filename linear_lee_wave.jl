using Roots
using CairoMakie
using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Grids: with_number_type
using Oceananigans.Forcings: MultipleForcings
using Oceananigans.Utils: launch!
using Oceananigans.Operators
using Oceananigans.Architectures: architecture
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA

const k = π / 2
const N² = 0.4
const U = 0.1
const h₀ = 0.1

#%%
const Lx = 8
const Lz = 8

const Nx = 4096
const Nz = 4096
# const Nx = 2048
# const Nz = 2048

advection = WENO(order=9)
# advection = Centered()

if advection isa WENO
    advection_str = "WENO"
else
    advection_str = "Centered"
end

k_str = k == π ? "pi" : string(k)
FILE_DIR = "./Data/linear_lee_wave_$(advection_str)_norightsponge_k_$(k_str)_N2_$(N²)_U_$(U)_h0_$(h₀)_Lx_$(Lx)_Lz_$(Lz)_Nx_$(Nx)_Nz_$(Nz)"
mkpath(FILE_DIR)

bottom_topography(x) = h₀ * (cos(k*x) + 1)

grid = RectilinearGrid(GPU(), Float64,
                        size = (Nx, Nz), 
                        halo = (6, 6),
                        x = (-Lx/2, Lx/2),
                        z = (0, Lz),
                        topology = (Periodic, Flat, Bounded))

grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom_topography))

reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)

preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100; preconditioner)

uᵢ(x, z) = -U * z
bᵢ(x, z) = N² * z

u_target(x, z, t) = uᵢ(x, z)
b_target(x, z, t) = bᵢ(x, z)

initial_Δt = (Lx / Nx) / U / 5
damping_rate = 1 / (initial_Δt * 20)

top_mask(x, z) = exp(-(z - Lz)^2 / (2 * (Lz/10)^2))
# right_mask(x, z) = exp(-(x - Lx/2)^2 / (2 * (Lx/10)^2))

u_top_sponge = Relaxation(rate=damping_rate, mask=top_mask, target=u_target)
vw_top_sponge = Relaxation(rate=damping_rate, mask=top_mask)
b_top_sponge = Relaxation(rate=damping_rate, mask=top_mask, target=b_target)

# u_right_sponge = Relaxation(rate=damping_rate, mask=right_mask, target=u_target)
# vw_right_sponge = Relaxation(rate=damping_rate, mask=right_mask)
# b_right_sponge = Relaxation(rate=damping_rate, mask=right_mask, target=b_target)

# u_forcings = MultipleForcings(u_top_sponge, u_right_sponge)
# vw_forcings = MultipleForcings(vw_top_sponge, vw_right_sponge)
# b_forcings = MultipleForcings(b_top_sponge, b_right_sponge)

model = NonhydrostaticModel(; grid, pressure_solver,
                              advection = advection,
                              tracers = :b,
                              buoyancy = BuoyancyTracer(),
                              forcing = (u=u_top_sponge, v=vw_top_sponge, w=vw_top_sponge, b=b_top_sponge))
                            #   forcing = (u=u_forcings, v=vw_forcings, w=vw_forcings, b=b_forcings))

set!(model, u=uᵢ, b=bᵢ)

stop_time = 200
simulation = Simulation(model; Δt=initial_Δt, stop_time=stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(10))

u, v, w = model.velocities
b = model.tracers.b

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

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("Iter: %d, time: %s, Δt: %s, Poisson iters: %d",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), pressure_iters)

    compute_flow_divergence!(d, sim.model)

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max b: %6.3e, max d: %6.3e, max pressure: %6.3e, mean pressure: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.b),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    mean(sim.model.pressures.pNHS),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

model_outputs = (; u, v, w, b, d, pNHS = model.pressures.pNHS)

simulation.output_writers[:jld2] = JLD2Writer(model, model_outputs;
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(0.25),
                                                          with_halos = true,
                                                          overwrite_existing = true)

run!(simulation)
#%%
u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u", backend=OnDisk())
v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "v", backend=OnDisk())
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w", backend=OnDisk())
b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b", backend=OnDisk())
p_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "pNHS", backend=OnDisk())
d_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "d", backend=OnDisk())
#%%
Nt = length(u_data.times)
times = u_data.times

fig = Figure(size=(2000, 1000))

n = Observable(1)

btitlestr = @lift @sprintf("Buoyancy at t = %.2f", times[$n])
utitlestr = @lift @sprintf("Horizontal velocity at t = %.2f", times[$n])
wtitlestr = @lift @sprintf("Vertical velocity at t = %.2f", times[$n])

axb = Axis(fig[1, 1], title=btitlestr)
axu = Axis(fig[1, 2], title=utitlestr)
axw = Axis(fig[1, 3], title=wtitlestr)

bn = @lift interior(b_data[$n], :, 1, :)
un = @lift interior(u_data[$n], :, 1, :)
wn = @lift interior(w_data[$n], :, 1, :)

blim = extrema(interior(b_data[Nt]))
ulim = extrema(interior(u_data[Nt]))
wlim = (-maximum(abs, interior(w_data[Nt])), maximum(abs, interior(w_data[Nt]))) ./ 2

hmb = heatmap!(axb, bn, colormap=:turbo, colorrange=blim)
hmu = heatmap!(axu, un, colormap=:turbo, colorrange=ulim)
hmw = heatmap!(axw, wn, colormap=:balance, colorrange=wlim)

Colorbar(fig[2, 1], hmb; label="Buoyancy", vertical=false, flipaxis=false)
Colorbar(fig[2, 2], hmu; label="u velocity", vertical=false, flipaxis=false)
Colorbar(fig[2, 3], hmw; label="w velocity", vertical=false, flipaxis=false)

CairoMakie.record(fig, "./$(FILE_DIR)/$(FILE_DIR).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

# for nn in 1:Nt
#     n[] = nn
#     save("./$(FILE_DIR)/buoyancy_t_$(nn).png", fig)
# end
#%%