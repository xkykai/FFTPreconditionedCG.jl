using JLD2
using Statistics
using CairoMakie
using Makie

filepath = "./reports/single_H100_timed_nogc.jld2"

Ns = [32, 64, 96, 128, 192, 256, 384, 512]

file = jldopen(filepath, "r")

precond_names = ["no", "FFT64", "FFT32", "MITgcm"]

median_times = Dict{String, Vector{Float64}}()
median_cg_iters = Dict{String, Vector{Float64}}()

for name in precond_names
    median_times[name] = zeros(length(Ns))
    median_cg_iters[name] = zeros(length(Ns))
    for (i, N) in enumerate(Ns)
        nsamples = length(file["$(N)/times/$(name)"])
        times = file["$(N)/times/$(name)"]
        median_times[name][i] = median([t.time for t in times])
        cg_iters = file["$(N)/cg_iters/$(name)"]
        median_cg_iters[name][i] = median(cg_iters)
    end
end

median_times["FFT_only"] = zeros(length(Ns))
for (i, N) in enumerate(Ns)
    nsamples = length(file["$(N)/times/FFTstep"])
    times = file["$(N)/times/FFTstep"]
    median_times["FFT_only"][i] = median([t.time for t in times])
end

#%%
colors = Makie.wong_colors();
linewidth = 3
fig = Figure(size=(1000, 500), fontsize=15)
axtime = Axis(fig[1, 1], xlabel="N³", ylabel="Median Time per Timestep (s)", yscale = log10, xscale=log2)
axiters = Axis(fig[1, 2], xlabel="N³", ylabel="Median CG Iterations per Timestep", yscale = log10, xscale=log2)
lines!(axtime, Ns, median_times["no"], label="No Preconditioner", linewidth=linewidth, color=colors[1])
lines!(axtime, Ns, median_times["FFT64"], label="FFT64 Preconditioner", linewidth=linewidth, color=colors[2])
lines!(axtime, Ns, median_times["FFT32"], label="FFT32 Preconditioner", linewidth=linewidth, color=colors[3])
lines!(axtime, Ns, median_times["MITgcm"], label="MITgcm Preconditioner", linewidth=linewidth, color=colors[4])
lines!(axtime, Ns, median_times["FFT_only"], label="FFT Only", linewidth=linewidth, color=colors[5])

lines!(axiters, Ns, median_cg_iters["no"], label="No Preconditioner", linewidth=linewidth, color=colors[1])
lines!(axiters, Ns, median_cg_iters["FFT64"], label="FFT64 Preconditioner", linewidth=linewidth, color=colors[2])
lines!(axiters, Ns, median_cg_iters["FFT32"], label="FFT32 Preconditioner", linewidth=linewidth, color=colors[3])
lines!(axiters, Ns, median_cg_iters["MITgcm"], label="MITgcm Preconditioner", linewidth=linewidth, color=colors[4])

Legend(fig[2, :], axtime, orientation=:horizontal)

display(fig)
save("./Output/benchmark_rough_rayleigh_benard_single_H100_nogc.png", fig, px_per_unit=4)
#%%