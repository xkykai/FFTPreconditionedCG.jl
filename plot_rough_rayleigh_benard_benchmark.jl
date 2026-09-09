using JLD2
using Statistics
using CairoMakie
using Makie

filepath = "./reports/single_H100.jld2"

Ns = [32, 64, 96, 128, 192, 256, 384, 512]

solver_names = ["FFT", "no", "FFT64", "FFT32", "MITgcm"]

labels = Dict("FFT" => "FFT Only",
              "no" => "No Preconditioner",
              "FFT64" => "FFT64 Preconditioner",
              "FFT32" => "FFT32 Preconditioner",
              "MITgcm" => "MITgcm Preconditioner")

nrepeats = 3

median_wall = Dict{String, Vector{Float64}}()
median_device = Dict{String, Vector{Float64}}()
median_iters = Dict{String, Vector{Float64}}()

file = jldopen(filepath, "r")

for name in solver_names
    median_wall[name] = zeros(length(Ns))
    median_device[name] = zeros(length(Ns))
    median_iters[name] = zeros(length(Ns))
    for (i, N) in enumerate(Ns)
        results = [file["$(N)/$(name)/$(r)"] for r in 1:nrepeats]
        median_wall[name][i] = median(vcat((r.wall for r in results)...))
        median_device[name][i] = median(vcat((r.device for r in results)...))
        median_iters[name][i] = median(vcat((r.iterations for r in results)...))
    end
end

close(file)

#%%
colors = Makie.wong_colors();
linewidth = 3
fig = Figure(size=(1400, 500), fontsize=15)
axtime = Axis(fig[1, 1], xlabel="N", ylabel="Median Time per Timestep (s)", yscale=log10, xscale=log2)
axiters = Axis(fig[1, 2], xlabel="N", ylabel="Median CG Iterations per Timestep", yscale=log10, xscale=log2)
axhost = Axis(fig[1, 3], xlabel="N", ylabel="Host Time Not Overlapped (%)", xscale=log2)

for (c, name) in enumerate(solver_names)
    lines!(axtime, Ns, median_wall[name], label=labels[name], linewidth=linewidth, color=colors[c])
    lines!(axhost, Ns, 100 .* (median_wall[name] .- median_device[name]) ./ median_wall[name],
           linewidth=linewidth, color=colors[c])
    name == "FFT" && continue
    lines!(axiters, Ns, median_iters[name], label=labels[name], linewidth=linewidth, color=colors[c])
end

Legend(fig[2, :], axtime, orientation=:horizontal)

display(fig)
save("./Output/benchmark_rough_rayleigh_benard_single_H100.png", fig, px_per_unit=4)
#%%
