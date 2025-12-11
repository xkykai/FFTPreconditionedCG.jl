using Printf
using Dates
using Oceananigans
using Oceananigans.Operators
using Oceananigans.Units
using Oceananigans.Solvers
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_field
using Oceananigans.Grids: with_number_type
using Random
using CairoMakie
using CUDA

# ---------------------------------------------------------------------- #
# Define Parameters
function setup_grid(Nx, Nz, Lx, Lz)
    arch = GPU()

    immersed_grid = begin
        underlying_grid = RectilinearGrid(
            arch,
            size = (Nx, Nz),
            x = (-Lx/2, Lx/2),
            z = (-Lz, 0),
            topology = (Periodic, Flat, Bounded),
            halo = (4, 4)
        )
    
        dx = Lx / Nx
    
        basin(x) = -Lz + 10 * exp(-x^2 / (2 * 16^2))
    
        ImmersedBoundaryGrid(
            underlying_grid,
            GridFittedBottom(basin)
        )
    end

    return immersed_grid
end

Nx = 256
Nz = 64
Lx = 128meters
Lz = 32meters

immersed_grid = setup_grid(Nx, Nz, Lx, Lz)
reduced_precision_grid = with_number_type(Float32, immersed_grid.underlying_grid)

# pressure_solver = nothing
# preconditioner = FFTBasedPoissonSolver(immersed_grid.underlying_grid)
preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
reltol = abstol = 1e-7 * 0.5^3^2
pressure_solver = ConjugateGradientPoissonSolver(immersed_grid, maxiter=100; reltol, abstol, preconditioner)

# advection = WENO(order=5)
advection = Centered()

function setup_model(grid, pressure_solver)
    diffusivity = 1.0e-5
    Pr = 1
    viscosity = Pr * diffusivity
    closure = ScalarDiffusivity(ν = viscosity, κ = diffusivity)

    model = NonhydrostaticModel(; grid,
                                  advection,
                                  tracers = (:T, :S),
                                  buoyancy = SeawaterBuoyancy(),
                                  closure,
                                  pressure_solver = pressure_solver)
    return model
end

function initial_conditions!(model)
    initial_T(x, z) = 20 + rand() * 1e-5
    initial_S(x, z) = 35

    set!(model, T = initial_T)
    set!(model, S = initial_S)
end

function setup_simulation(model)
    Δt = 1
    stop_time = 1days
    simulation = Simulation(model; Δt = Δt, stop_time = stop_time, minimum_relative_step = 1e-10)
    
    wizard = TimeStepWizard(max_change=1.05, max_Δt=5, cfl=0.6)

    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    wall_time = Ref(time_ns())

    function progress(sim)
        pressure_solver = sim.model.pressure_solver
    
        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end
    
        msg = @sprintf("Iter: %d, time: %s, Δt: %.4f, Poisson iters: %d",
                        iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)
    
        elapsed = 1e-9 * (time_ns() - wall_time[])
    
        u, v, w = sim.model.velocities
        d = Field(∂x(u) + ∂y(v) + ∂z(w))
        compute!(d)
    
        msg *= @sprintf(", max u: %6.3e, max w: %6.3e, max T: %6.3e, max S: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.velocities.w),
                        maximum(abs, sim.model.tracers.T),
                        maximum(abs, sim.model.tracers.S),
                        maximum(abs, d),
                        maximum(abs, sim.model.pressures.pNHS),
                        prettytime(elapsed))
    
        @info msg
        wall_time[] = time_ns()
    
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    
    u, v, w = model.velocities
    d = Field(∂x(u) + ∂z(w))

    T, S = model.tracers
    p = model.pressures.pNHS
    
    b = buoyancy_field(model)

    Tbar = Average(T, dims=(1, 2, 3))
    Sbar = Average(S, dims=(1, 2, 3))

    prefix = "gaussian_ridge"
    pressure_solver = model.pressure_solver

    if pressure_solver isa ConjugateGradientPoissonSolver
        prefix *= "_cg"
    else
        prefix *= "_fft"
    end

    outputs = (; u, v, w, T, S, b, d, p)
    averaged_outputs = (; Tbar, Sbar)

    OUTPUT_PATH = "./Data/$(prefix)_mwe_fields_constantS"
    simulation.output_writers[:jld2] = JLD2Writer(model, outputs,
                                                        schedule = TimeInterval(10minutes),
                                                        filename = "$(OUTPUT_PATH)_fields",
                                                        overwrite_existing = true,
                                                        with_halos = true)

    simulation.output_writers[:jld2_averaged] = JLD2Writer(model, averaged_outputs,
                                                                schedule = TimeInterval(10minutes),
                                                                filename = "$(OUTPUT_PATH)_averaged",
                                                                overwrite_existing = true,
                                                                with_halos = true)

    return simulation
end

model = setup_model(immersed_grid, pressure_solver)
initial_conditions!(model)
simulation = setup_simulation(model)

run!(simulation)

model_FFT = setup_model(immersed_grid, nothing)
initial_conditions!(model_FFT)
simulation_FFT = setup_simulation(model_FFT)
run!(simulation_FFT)
#%%
function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

#%%
T_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_fields.jld2", "T")
S_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_fields.jld2", "S")
b_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_fields.jld2", "b")
d_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_fields.jld2", "d")
p_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_fields.jld2", "p")

T_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_fields.jld2", "T")
S_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_fields.jld2", "S")
b_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_fields.jld2", "b")
d_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_fields.jld2", "d")
p_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_fields.jld2", "p")
#%%
Tbar_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_averaged.jld2", "Tbar")
Sbar_data_fft = FieldTimeSeries("./Data/gaussian_ridge_fft_mwe_fields_constantS_averaged.jld2", "Sbar")

Tbar_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_averaged.jld2", "Tbar")
Sbar_data_cg = FieldTimeSeries("./Data/gaussian_ridge_cg_mwe_fields_constantS_averaged.jld2", "Sbar")
#%%
times = T_data_fft.times
Nt = length(times)

xC = T_data_fft.grid.underlying_grid.xᶜᵃᵃ[1:Nx]
zC = T_data_fft.grid.underlying_grid.z.cᵃᵃᶜ[1:Nz]

#%%
fig = Figure(size=(2000, 1200))
axT_fft = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "Temperature (FFT solver)")
axS_fft = Axis(fig[1, 2], xlabel = "x (m)", ylabel = "z (m)", title = "Salinity (FFT solver)")
axd_fft = Axis(fig[1, 3], xlabel = "x (m)", ylabel = "z (m)", title = "Divergence (FFT solver)")
axp_fft = Axis(fig[1, 4], xlabel = "x (m)", ylabel = "z (m)", title = "Pressure (FFT solver)")
axb_fft = Axis(fig[1, 5], xlabel = "x (m)", ylabel = "z (m)", title = "Buoyancy (FFT solver)")

axT_cg = Axis(fig[3, 1], xlabel = "x (m)", ylabel = "z (m)", title = "Temperature (FFT-preconditioned CG solver)")
axS_cg = Axis(fig[3, 2], xlabel = "x (m)", ylabel = "z (m)", title = "Salinity (FFT-preconditioned CG solver)")
axd_cg = Axis(fig[3, 3], xlabel = "x (m)", ylabel = "z (m)", title = "Divergence (FFT-preconditioned CG solver)")
axp_cg = Axis(fig[3, 4], xlabel = "x (m)", ylabel = "z (m)", title = "Pressure (FFT-preconditioned CG solver)")
axb_cg = Axis(fig[3, 5], xlabel = "x (m)", ylabel = "z (m)", title = "Buoyancy (FFT-preconditioned CG solver)")

axTbar = Axis(fig[5, 1:2], xlabel = "time (days)", ylabel = "T (°C)", title = "Domain-averaged temperature")
axSbar = Axis(fig[5, 4:5], xlabel = "time (days)", ylabel = "S (psu)", title = "Domain-averaged salinity")

n = Observable(Nt)

T_fftₙ = @lift interior(T_data_fft[$n], :, 1, :)
S_fftₙ = @lift interior(S_data_fft[$n], :, 1, :)
d_fftₙ = @lift interior(d_data_fft[$n], :, 1, :)
p_fftₙ = @lift interior(p_data_fft[$n], :, 1, :)
b_fftₙ = @lift interior(b_data_fft[$n], :, 1, :)

T_cgₙ = @lift interior(T_data_cg[$n], :, 1, :)
S_cgₙ = @lift interior(S_data_cg[$n], :, 1, :)
d_cgₙ = @lift interior(d_data_cg[$n], :, 1, :)
p_cgₙ = @lift interior(p_data_cg[$n], :, 1, :)
b_cgₙ = @lift interior(b_data_cg[$n], :, 1, :)

Tbar_cg = interior(Tbar_data_cg, 1, 1, 1, :)
Sbar_cg = interior(Sbar_data_cg, 1, 1, 1, :)

Tbar_fft = interior(Tbar_data_fft, 1, 1, 1, :)
Sbar_fft = interior(Sbar_data_fft, 1, 1, 1, :)

timeₙ = @lift [times[$n] / 1days]

timeframe_lim = Nt
Tlim_FFT = (find_min(interior(T_data_fft[timeframe_lim])[interior(T_data_fft[timeframe_lim]) .!= 0]), find_max(interior(T_data_fft[timeframe_lim])[interior(T_data_fft[timeframe_lim]) .!= 0]))
Slim_FFT = (find_min(interior(S_data_fft[timeframe_lim])[interior(S_data_fft[timeframe_lim]) .!= 0]), find_max(interior(S_data_fft[timeframe_lim])[interior(S_data_fft[timeframe_lim]) .!= 0]))
blim_FFT = (find_min(interior(b_data_fft[timeframe_lim])[interior(b_data_fft[timeframe_lim]) .!= 0]), find_max(interior(b_data_fft[timeframe_lim])[interior(b_data_fft[timeframe_lim]) .!= 0]))
dlim_FFT = (-maximum(abs, interior(d_data_fft[timeframe_lim])[interior(d_data_fft[timeframe_lim]) .!= 0]), maximum(abs, interior(d_data_fft[timeframe_lim])[interior(d_data_fft[timeframe_lim]) .!= 0]))
plim_FFT = (-maximum(abs, interior(p_data_fft)), maximum(abs, interior(p_data_fft))) ./ 4

Tlim_cg = (find_min(interior(T_data_cg[timeframe_lim])[interior(T_data_cg[timeframe_lim]) .!= 0]) - 1e-4, find_max(interior(T_data_cg[timeframe_lim])[interior(T_data_cg[timeframe_lim]) .!= 0]) + 1e-4)
Slim_cg = (find_min(interior(S_data_cg[timeframe_lim])[interior(S_data_cg[timeframe_lim]) .!= 0]) - 1e-4, find_max(interior(S_data_cg[timeframe_lim])[interior(S_data_cg[timeframe_lim]) .!= 0]) + 1e-4)
blim_cg = (find_min(interior(b_data_cg[timeframe_lim])[interior(b_data_cg[timeframe_lim]) .!= 0]) - 1e-4, find_max(interior(b_data_cg[timeframe_lim])[interior(b_data_cg[timeframe_lim]) .!= 0]) + 1e-4)
dlim_cg = (-maximum(abs, interior(d_data_cg[timeframe_lim])[interior(d_data_cg[timeframe_lim]) .!= 0]) - 1e-4, maximum(abs, interior(d_data_cg[timeframe_lim])[interior(d_data_cg[timeframe_lim]) .!= 0]) + 1e-4)
plim_cg = (-maximum(abs, interior(p_data_cg)) - 1e-4, maximum(abs, interior(p_data_cg)) + 1e-4)

Tbarlim = (find_min(interior(Tbar_data_fft)) - 1e-4, find_max(interior(Tbar_data_fft)) + 1e-4)
Sbarlim = (find_min(interior(Sbar_data_fft)) - 1e-4, find_max(interior(Sbar_data_fft)) + 1e-4)

colormap_symmetric = :balance
colormap_asymmetric = :turbo

hmT_FFT = heatmap!(axT_fft, xC, zC, T_fftₙ, colorrange = Tlim_FFT, colormap = colormap_asymmetric)
hmS_FFT = heatmap!(axS_fft, xC, zC, S_fftₙ, colorrange = Slim_FFT, colormap = Reverse(colormap_asymmetric))
hmd_FFT = heatmap!(axd_fft, xC, zC, d_fftₙ, colorrange = dlim_FFT, colormap = colormap_symmetric)
hmp_FFT = heatmap!(axp_fft, xC, zC, p_fftₙ, colorrange = plim_FFT, colormap = colormap_symmetric)
hmb_FFT = heatmap!(axb_fft, xC, zC, b_fftₙ, colorrange = blim_FFT, colormap = colormap_asymmetric)

hmT_cg = heatmap!(axT_cg, xC, zC, T_cgₙ, colorrange = Tlim_cg, colormap = colormap_asymmetric)
hmS_cg = heatmap!(axS_cg, xC, zC, S_cgₙ, colorrange = Slim_cg, colormap = Reverse(colormap_asymmetric))
hmd_cg = heatmap!(axd_cg, xC, zC, d_cgₙ, colorrange = dlim_cg, colormap = colormap_symmetric)
hmp_cg = heatmap!(axp_cg, xC, zC, p_cgₙ, colorrange = plim_cg, colormap = colormap_symmetric)
hmb_cg = heatmap!(axb_cg, xC, zC, b_cgₙ, colorrange = blim_cg, colormap = colormap_asymmetric)

Colorbar(fig[2, 1], hmT_FFT, label = "°C", vertical=false, flipaxis=false)
Colorbar(fig[2, 2], hmS_FFT, label = "psu", vertical=false, flipaxis=false)
Colorbar(fig[2, 3], hmd_FFT, label = "s⁻¹", vertical=false, flipaxis=false)
Colorbar(fig[2, 4], hmp_FFT, label = "Pa", vertical=false, flipaxis=false)
Colorbar(fig[2, 5], hmb_FFT, label = "m²/s²", vertical=false, flipaxis=false)

Colorbar(fig[4, 1], hmT_cg, label = "°C", vertical=false, flipaxis=false)
Colorbar(fig[4, 2], hmS_cg, label = "psu", vertical=false, flipaxis=false)
Colorbar(fig[4, 3], hmd_cg, label = "s⁻¹", vertical=false, flipaxis=false)
Colorbar(fig[4, 4], hmp_cg, label = "Pa", vertical=false, flipaxis=false)
Colorbar(fig[4, 5], hmb_cg, label = "m²/s²", vertical=false, flipaxis=false)

lines!(axTbar, times ./ 1days, Tbar_cg, label="PCG")
lines!(axSbar, times ./ 1days, Sbar_cg, label="PCG")

lines!(axTbar, times ./ 1days, Tbar_fft, label="FFT")
lines!(axSbar, times ./ 1days, Sbar_fft, label="FFT")

vlines!(axTbar, timeₙ, color = :black)
vlines!(axSbar, timeₙ, color = :black)

ylims!(axTbar, Tbarlim)
ylims!(axSbar, Sbarlim)

axislegend(axTbar)

display(fig)

# save("./cg_fft_gaussian_ridge_mwe_fields_constantS.pdf", fig)
CairoMakie.record(fig, "./Output/cg_fft_gaussian_ridge_mwe_fields_constantS_centered.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
    n[] = nn
end
#%%
residual = Array(interior(model.pressure_solver.conjugate_gradient_solver.residual))
rlim = (-maximum(abs, residual), maximum(abs, residual)) ./ 2
fig = Figure(size=(1000, 700))
ax = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "Pressure CG solver residual")
hm = heatmap!(ax, xC, zC, Array(interior(model.pressure_solver.conjugate_gradient_solver.residual))[:, 1, :], colorrange = rlim, colormap=:balance)
Colorbar(fig[1, 2], hm, label = "Pa")
display(fig)
save("./cg_fft_mwe_fields_residual.png", fig, px_per_unit = 4)
#%%