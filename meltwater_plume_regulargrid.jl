using CairoMakie
using Oceananigans
using Printf
using JLD2
using Oceananigans.Grids: xnodes, znodes
using Oceananigans.Utils: launch!
using Oceananigans.Operators
using Oceananigans.Architectures: architecture
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA

const L = 1
const Lx = L * 2
const Lz = L * 4
const Ra = 1e10
const N² = 1
const B = 1
const Pr = 1

const ν = sqrt(B^4 / (N²)^3 / Ra * Pr)
const κ = ν / Pr

const Nx = 1024
const Nz = 2048

advection = Centered()
closure = ScalarDiffusivity(; ν, κ)

if advection isa WENO
    advection_str = "WENO"
elseif advection isa Centered
    advection_str = "Centered"
end

filename = "meltwater_plume_$(advection_str)_Ra_$(Ra)_Pr_$(Pr)_Lx_$(Lx)_Lz_$(Lz)_Nx_$(Nx)_Nz_$(Nz)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)
@info "Data will be saved to $(FILE_DIR)"

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Nz), 
                       x = (0, Lx),
                       z = (0, Lz),
                       topology = (Bounded, Flat, Periodic))

no_slip_bc = ValueBoundaryCondition(0)

const τ = 10 / sqrt(N²)
@inline b_timeramp(j, k, grid, clock, model_fields, p) = p.B * tanh(clock.time / p.τ)

b_west_bc = ValueBoundaryCondition(b_timeramp, discrete_form=true, parameters=(; B, τ))

w_bcs = FieldBoundaryConditions(no_slip_bc)
b_bcs = FieldBoundaryConditions(west = b_west_bc, east = ValueBoundaryCondition(0))
c_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(B), east = ValueBoundaryCondition(0))

b_forcing_func(x, z, t, w, N²) = -w * N²
b_forcing = Forcing(b_forcing_func, field_dependencies=:w, parameters=N²)

model = NonhydrostaticModel(; grid,
                              tracers = (:b, :c),
                              buoyancy = BuoyancyTracer(),
                              advection,
                              closure,
                              boundary_conditions = (w = w_bcs, b = b_bcs, c = c_bcs),
                              forcing = (; b = b_forcing),
                              hydrostatic_pressure_anomaly = nothing)

b₁(x, z) = rand() * 1e-5
set!(model, b = b₁)

stop_time = 500 / sqrt(N²)
Δt = min((Lz / Nz) / B, (Lx / Nx)^2 / max(ν, κ)) / 5
simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(10))

u, v, w = model.velocities
b, c = model.tracers.b, model.tracers.c

function progress(sim)
    msg = @sprintf("Iter: %d, time: %s, Δt: %s",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt))

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max b: %6.3e, max c: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.b),
                    maximum(abs, sim.model.tracers.c),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

const l = (4 * ν * κ / N²)^(1/2)
Nu = Field(Average(-∂x(b) / (B / l), dims=(2, 3)))
bbar = Field(Average(b, dims=(2, 3)))

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, w, b, c);
                                              filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                              schedule = TimeInterval(5),
                                              with_halos = true,
                                              overwrite_existing = true)

simulation.output_writers[:averaged] = JLD2Writer(model, (; Nu, b = bbar);
                                              filename = joinpath(FILE_DIR, "averaged_fields.jld2"),
                                              schedule = AveragedTimeInterval(100, window=100),
                                              indices = (1, 1, 1),
                                              with_halos = true,
                                              overwrite_existing = true)

run!(simulation)
#%%
u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u")
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b")
c_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "c")
#%%
Nt = length(u_data.times)
times = u_data.times

xC = xnodes(u_data.grid, Center())
xF = xnodes(u_data.grid, Face())
zC = znodes(u_data.grid, Center())
zF = znodes(u_data.grid, Face())
#%%
fig = Figure(size=(2000, 1000), fontsize=20)

n = Observable(Nt)

axu = Axis(fig[1, 1]; title = "u velocity", xlabel = "x", ylabel = "z")
axw = Axis(fig[1, 2]; title = "w velocity", xlabel = "x", ylabel = "z")
axb = Axis(fig[1, 3]; title = "buoyancy", xlabel = "x", ylabel = "z")
axc = Axis(fig[1, 4]; title = "concentration", xlabel = "x", ylabel = "z")

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
bₙ = @lift interior(b_data[$n], :, 1, :)
cₙ = @lift interior(c_data[$n], :, 1, :)

ulim = (-maximum(abs, interior(u_data[Nt])), maximum(abs, interior(u_data[Nt]))) ./ 2
wlim = (-maximum(abs, interior(w_data[Nt])), maximum(abs, interior(w_data[Nt]))) ./ 2
blim = (-maximum(abs, interior(b_data[Nt])), maximum(abs, interior(b_data[Nt]))) ./ 2
clim = (0, 1)

hmu = heatmap!(axu, xF, zC, uₙ, colormap=:balance, colorrange=ulim)
hmw = heatmap!(axw, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
hmb = heatmap!(axb, xC, zC, bₙ, colormap=:balance, colorrange=blim)
hmc = heatmap!(axc, xC, zC, cₙ, colormap=:plasma, colorrange=clim)

Colorbar(fig[2, 1], hmu; label = "u", vertical=false, flipaxis=false)
Colorbar(fig[2, 2], hmw; label = "w", vertical=false, flipaxis=false)
Colorbar(fig[2, 3], hmb; label = "buoyancy", vertical=false, flipaxis=false)
Colorbar(fig[2, 4], hmc; label = "concentration", vertical=false, flipaxis=false)

time_str = @lift @sprintf("t = %.2f s", times[$n])
Label(fig[0, :], time_str, fontsize=20)

CairoMakie.record(fig, "./$(FILE_DIR)/$(filename).mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
#%%
Nu_data = FieldTimeSeries("$(FILE_DIR)/averaged_fields.jld2", "Nu")
Nu = interior(Nu_data, 1, 1, 1, :)
times = Nu_data.times

xF = xnodes(Nu_data.grid, Face())
fig = Figure()
ax = Axis(fig[1, 1]; title = "Nusselt number profile", xlabel = "t", ylabel = "Nu")
lines!(ax, times, Nu)
save("./$(FILE_DIR)/$(filename)_nusselt.png", fig, px_per_unit=4)
#%%