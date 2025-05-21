using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, compute_laplacian!
using Oceananigans.Grids: with_number_type
using Statistics
using CairoMakie

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, z) = - exp(-((x - x₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid(Nx, Nz)
    grid = RectilinearGrid(GPU(), Float64,
                        size = (Nx, Nz), 
                        halo = (6, 6),
                        x = (0, 1),
                        z = (0, 1),
                        topology = (Bounded, Flat, Bounded))

    slope(x) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 10

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

Nx = Nz = 128
grid = setup_grid(Nx, Nz)
reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)

preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100; preconditioner)

model = setup_model(grid, pressure_solver)

stop_time = 10
simulation = Simulation(model; Δt=1e-4, stop_time=stop_time, minimum_relative_step = 1e-10)

time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, min_Δt=1e-4, max_Δt=1)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(10))

u, v, w = model.velocities
b = model.tracers.b

d = Field(∂x(u) + ∂y(v) + ∂z(w))
wall_time = Ref(time_ns())

B = Field(Integral(b))

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("Iter: %d, time: %s, Δt: %6.3e, Poisson iters: %d",
                    iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    compute!(d)

    msg *= @sprintf(", max u: %6.3e, max w: %6.3e, max b: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.b),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    prettytime(elapsed))

    @info msg
    wall_time[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; p=model.pressures.pNHS, d, B))

filename = "./Data/staircase_convection_2D_fields.jld2"
simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = filename,
                                              schedule = TimeInterval(0.1),
                                              overwrite_existing = true)

run!(simulation)

#%%
bt = FieldTimeSeries(filename, "b")
ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")
pt = FieldTimeSeries(filename, "p")
δt = FieldTimeSeries(filename, "d")
times = bt.times
Nt = length(times)

Bt = FieldTimeSeries(filename, "B")
#%%
fig = Figure(size=(1200, 1200))

n = Observable(1)

B₀ = sum(interior(bt[1], :, 1, :)) / (Nx * Nz)
btitlestr = @lift @sprintf("Buoyancy at t = %.2f", times[$n])
utitlestr = @lift @sprintf("Horizontal velocity at t = %.2f", times[$n])
wtitlestr = @lift @sprintf("Vertical velocity at t = %.2f", times[$n])

axb = Axis(fig[1, 1], title=btitlestr)
axu = Axis(fig[1, 2], title=utitlestr)
axw = Axis(fig[1, 3], title=wtitlestr)
axp = Axis(fig[2, 1], title="Pressure")
axd = Axis(fig[2, 2], title="Divergence")
axt = Axis(fig[3, 1:3], xlabel="Time", ylabel="Fractional remaining tracer")

bn = @lift interior(bt[$n], :, 1, :)
un = @lift interior(ut[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)
pn = @lift interior(pt[$n], :, 1, :)
δn = @lift interior(δt[$n], :, 1, :)

ulim = maximum(abs, ut) / 2
wlim = maximum(abs, wt) / 2
plim = maximum(abs, pt) / 2
δlim = 1e-8

heatmap!(axb, bn, colormap=:balance, colorrange=(-0.5, 0.5))
heatmap!(axu, un, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw, wn, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axp, pn, colormap=:balance, colorrange=(-plim, plim))
heatmap!(axd, δn, colormap=:balance, colorrange=(-δlim, δlim))

ΔB = Bt.data[1, 1, 1, :] .- Bt.data[1, 1, 1, 1]
t = @lift times[$n]
lines!(axt, times, ΔB)
vlines!(axt, t, color=:black)
# display(fig)

CairoMakie.record(fig, "./Output/staircase_convection_2D.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end
