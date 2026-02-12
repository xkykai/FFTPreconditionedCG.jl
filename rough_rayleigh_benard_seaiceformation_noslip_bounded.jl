using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Models: buoyancy_field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Grids: with_number_type, xnodes, znodes
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA
using CairoMakie
using NaNStatistics
using ArgParse
using Glob

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--solver"
        help = "Type of pressure solver to use (FFT or CG)"
        arg_type = String
        default = "FFT"
      "--N"
        help = "Grid size in each direction (Nx = Nz = N)"
        arg_type = Int
        default = 1024
      "--Ra"
        help = "Rayleigh number"
        arg_type = Float64
        default = 1e8
    end
    return parse_args(s)
end

args = parse_commandline()
solver_type = args["solver"]
N = args["N"]

arch = GPU()

const Ra = args["Ra"]
const g = 1
const α = 1
const β = 4
const Pr = 1
const ΔT = -1
const ΔS = 1
const H = 1
const Δb = (-α*ΔT + β*ΔS) * g

const ν = sqrt(Δb * g * H^3 * Pr / Ra)
const κ = ν / Pr
const Lx = 1
const Lz = 1
const Nx = N
const Nz = N

closure = ScalarDiffusivity(ν=ν, κ=κ)

equation_of_state = LinearEquationOfState(thermal_expansion=α, haline_contraction=β)
buoyancy = SeawaterBuoyancy(; gravitational_acceleration=g, equation_of_state)

#####
##### Model setup
#####

@inline function local_roughness_top(η, η₀, h)
    if η > η₀ - h && η <= η₀
        return -η - h + η₀
    elseif η > η₀ && η <= η₀ + h
        return η - h - η₀
    else
        return 0
    end
end

grid = RectilinearGrid(arch, Float64,
                        size = (Nx, Nz), 
                        halo = (6, 6),
                        x = (0, Lx),
                        z = (0, Lz),
                        topology = (Bounded, Flat, Bounded))

const Nr = 16 # number of roughness elements
const hx = Lx / Nr / 2
const x₀s = hx:2hx:Lx-hx

@inline function roughness_top(x, z)
    z_rough_x = sum([local_roughness_top(x, x₀, hx) for x₀ in x₀s])

    return z >= z_rough_x + Lz
end

@inline mask(x, z) = roughness_top(x, z)
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(mask))

if solver_type == "FFT"
    pressure_solver = FFTBasedPoissonSolver(grid.underlying_grid)
    pressure_solver_str = solver_type
elseif solver_type == "CG"
    reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
    preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
    pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100; preconditioner)
    pressure_solver_str = solver_type
end

filename = "rough_RB_seaiceformation_noslip_bounded_Ra_$(Ra)_Pr_$(Pr)_Nr_$(Nr)_Lx_$(Lx)_Lz_$(Lz)_Nx_$(Nx)_Nz_$(Nz)_$(pressure_solver_str)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)

const T_top = 0
const T_bottom = 1

const S_top = 1
const S_bottom = 0

@inline function rayleigh_benard_T(x, z, t)
    above_centerline = z > 1 / 2
    return ifelse(above_centerline, T_top, T_bottom)
end

@inline function rayleigh_benard_S(x, z, t)
    above_centerline = z > 1 / 2
    return ifelse(above_centerline, S_top, S_bottom)
end

no_slip_bc = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(top=no_slip_bc, bottom=no_slip_bc, immersed=no_slip_bc)
v_bcs = FieldBoundaryConditions(top=no_slip_bc, bottom=no_slip_bc, immersed=no_slip_bc, east=no_slip_bc, west=no_slip_bc)
w_bcs = FieldBoundaryConditions(immersed=no_slip_bc, east=no_slip_bc, west=no_slip_bc)

T_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(T_top), bottom=ValueBoundaryCondition(T_bottom),
                                immersed=ValueBoundaryCondition(rayleigh_benard_T))
S_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(S_top), bottom=ValueBoundaryCondition(S_bottom),
                                immersed=ValueBoundaryCondition(rayleigh_benard_S))

boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

model = NonhydrostaticModel(; grid, pressure_solver,
                              advection = Centered(),
                              closure,
                              tracers = (:T, :S, :c),
                              buoyancy,
                              boundary_conditions)

Tᵢ(x, z) = T_bottom + rand() * 1e-5
Sᵢ(x, z) = S_bottom + rand() * 1e-5
c₁(x, z) = 1

set!(model, T=Tᵢ, c=c₁, S=Sᵢ)

stop_time = 2000
advective_Δt = (Lz / Nz) / Δb
diffusive_Δt = min((Lx / Nx)^2, Lz/Nz^2) / max(ν, κ)
Δt = min(advective_Δt, diffusive_Δt) / 10

simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, max_Δt=diffusive_Δt / 5)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
T, S, c = model.tracers.T, model.tracers.S, model.tracers.c
b = buoyancy_field(model)

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

    msg *= @sprintf(", max u: %6.3e, max w: %6.3e, max T: %6.3e, max S: %6.3e, max d: %6.3e, max pressure: %6.3e, mean pressure: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.T),
                    maximum(abs, sim.model.tracers.S),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    mean(sim.model.pressures.pNHS),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

wb = Field(w * b)
∂b∂z = α * (T_top - T_bottom) / Lz - β * (S_top - S_bottom) / Lz
Nu = Average(1 - wb / (κ * ∂b∂z), dims=(1, 2))

Tbar = Average(T, dims=(1, 2))
Sbar = Average(S, dims=(1, 2))
bbar = Average(b, dims=(1, 2))
KEbar = Average(0.5 * (u^2 + w^2), dims=(1, 2, 3))

p = model.pressures.pNHS + model.pressures.pHY′

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, w, T, S, c, b, d, p);
                                              filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                              schedule = TimeInterval(50),
                                              with_halos = true,
                                              overwrite_existing = true)

simulation.output_writers[:averaged] = JLD2Writer(model, (; T = Tbar, S = Sbar, b = bbar, Nu);
                                              filename = joinpath(FILE_DIR, "averaged_fields.jld2"),
                                              schedule = AveragedTimeInterval(1000, window=1000),
                                              with_halos = false,
                                              overwrite_existing = true)

simulation.output_writers[:KE] = JLD2Writer(model, (; KE = KEbar);
                                              filename = joinpath(FILE_DIR, "KE_fields.jld2"),
                                              schedule = AveragedTimeInterval(50, window=50),
                                              with_halos = false,
                                              overwrite_existing = true)

simulation.output_writers[:checkpoint] = Checkpointer(model;
                                                      dir = FILE_DIR,
                                                      schedule = TimeInterval(500))

checkpoint_files = glob("checkpoint*.jld2", FILE_DIR)
if !isempty(checkpoint_files)
    @info "Found checkpoint files, resuming from checkpoint"
    run!(simulation, pickup=true)
else
    @info "No checkpoint files found, starting fresh simulation"
    run!(simulation)
end

checkpoint_files = glob("checkpoint*.jld2", FILE_DIR)
if !isempty(checkpoint_files)
    @info "Simulation completed successfully, removing $(length(checkpoint_files)) checkpoint file(s)"
    for f in checkpoint_files
        rm(f)
    end
end
#%%
u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u")
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
T_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "T")
S_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "S")
d_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "d")
p_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "p")

Nt = length(u_data.times)
times = u_data.times

xC = xnodes(u_data.grid, Center())
xF = xnodes(u_data.grid, Face())
zC = znodes(u_data.grid, Center())
zF = znodes(u_data.grid, Face())

for i in 1:Nt
    mask_immersed_field!(u_data[i], NaN)
    mask_immersed_field!(w_data[i], NaN)
    mask_immersed_field!(T_data[i], NaN)
    mask_immersed_field!(S_data[i], NaN)
    mask_immersed_field!(d_data[i], NaN)
    mask_immersed_field!(p_data[i], NaN)
end

fig = Figure(size=(1800, 1000), fontsize=20)

n = Observable(Nt)

axu = Axis(fig[1, 1]; title = "u velocity", xlabel = "x", ylabel = "z")
axw = Axis(fig[1, 3]; title = "w velocity", xlabel = "x", ylabel = "z")
axT = Axis(fig[2, 1]; title = "temperature", xlabel = "x", ylabel = "z")
axS = Axis(fig[2, 3]; title = "salinity", xlabel = "x", ylabel = "z")
axd = Axis(fig[1, 5]; title = "flow divergence", xlabel = "x", ylabel = "z")
axp = Axis(fig[2, 5]; title = "pressure", xlabel = "x", ylabel = "z")

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
Tₙ = @lift interior(T_data[$n], :, 1, :)
Sₙ = @lift interior(S_data[$n], :, 1, :)
dₙ = @lift interior(d_data[$n], :, 1, :)
pₙ = @lift interior(p_data[$n], :, 1, :)

ulim = (-nanmaximum(abs.(interior(u_data[Nt]))), nanmaximum(abs.(interior(u_data[Nt])))) ./ 2
wlim = (-nanmaximum(abs.(interior(w_data[Nt]))), nanmaximum(abs.(interior(w_data[Nt])))) ./ 2
Tlim = (nanminimum(interior(T_data[Nt])), nanmaximum(interior(T_data[Nt])))
Slim = (nanminimum(interior(S_data[Nt])) - 1e-4, nanmaximum(interior(S_data[Nt])) + 1e-4)
dlim = (-nanmaximum(abs.(interior(d_data[Nt]))), nanmaximum(abs.(interior(d_data[Nt])))) ./ 2
plim = (-nanmaximum(abs.(interior(p_data[Nt]))), nanmaximum(abs.(interior(p_data[Nt])))) ./ 2

hmu = heatmap!(axu, xF, zC, uₙ, colormap=:balance, colorrange=ulim)
hmw = heatmap!(axw, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
hmT = heatmap!(axT, xC, zC, Tₙ, colormap=:turbo, colorrange=Tlim)
hmS = heatmap!(axS, xC, zC, Sₙ, colormap=:turbo, colorrange=Slim)
hmd = heatmap!(axd, xC, zC, dₙ, colormap=:balance, colorrange=dlim)
hmp = heatmap!(axp, xC, zC, pₙ, colormap=:balance, colorrange=plim)

Colorbar(fig[1, 2], hmu; label = "u")
Colorbar(fig[1, 4], hmw; label = "w")
Colorbar(fig[2, 2], hmT; label = "T")
Colorbar(fig[2, 4], hmS; label = "S")
Colorbar(fig[1, 6], hmd; label = "d")
Colorbar(fig[2, 6], hmp; label = "p")

time_str = @lift @sprintf("t = %.2f s", times[$n])
Label(fig[0, :], time_str, fontsize=20)

display(fig)
CairoMakie.record(fig, "./$(FILE_DIR)/$(filename).mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
#%%
Nu_data = FieldTimeSeries("$(FILE_DIR)/averaged_fields.jld2", "Nu")
Nt_averaged = length(Nu_data.times)

fig = Figure()
ax = Axis(fig[1, 1], title="Nu", xlabel="Nu", ylabel="z")
Nuₙ = interior(Nu_data[Nt_averaged], 1, 1, :)

lines!(ax, Nuₙ, znodes(Nu_data.grid, Face()))
display(fig)
save("./$(FILE_DIR)/Nu_profile.png", fig)
#%%
KE_data = FieldTimeSeries("$(FILE_DIR)/KE_fields.jld2", "KE")
Nt_averaged = length(KE_data.times)
fig = Figure()
ax = Axis(fig[1, 1], title="KE", xlabel="time", ylabel="KE")
times_averaged = KE_data.times
KE_values = interior(KE_data, 1, 1, 1, 1:Nt_averaged)
lines!(ax, times_averaged, KE_values)
display(fig)
save("./$(FILE_DIR)/KE_with_time.png", fig)
#%%
# FFT_DIR = "./Data/rough_RB_seaiceformation_Ra_1.0e10_Pr_1_Nr_8_Lx_1_Lz_1_Nx_4096_Nz_4096_FFT"
# CG_DIR = "./Data/rough_RB_seaiceformation_Ra_1.0e10_Pr_1_Nr_8_Lx_1_Lz_1_Nx_4096_Nz_4096_CG"

# Nu_data_FFT = FieldTimeSeries("$(FFT_DIR)/averaged_fields.jld2", "Nu")
# Nu_data_CG = FieldTimeSeries("$(CG_DIR)/averaged_fields.jld2", "Nu")
# Nt_averaged = length(Nu_data_FFT.times)
# #%%
# fig = Figure()
# ax = Axis(fig[1, 1], title="Nu comparison, Ra = $(Ra)", xlabel="Nu", ylabel="z")
# Nu_FFT = interior(Nu_data_FFT[Nt_averaged], 1, 1, :)
# Nu_CG = interior(Nu_data_CG[Nt_averaged], 1, 1, :)
# lines!(ax, Nu_FFT, znodes(Nu_data_FFT.grid, Face()), color=:blue, label="FFT")
# lines!(ax, Nu_CG, znodes(Nu_data_CG.grid, Face()), color=:red, label="CG")
# axislegend(ax, position=:rb)
# display(fig)
# save("./Output/Nu_profile_comparison_Ra_$(Ra)_N_4096.png", fig)
# # #%%