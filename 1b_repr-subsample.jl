using CSV, DataFrames
using Statistics, StatsBase

using CairoMakie
using MintsMakieRecipes

set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)



basepath = "./data/supervised"

@assert ispath(basepath)

target_names = readdir(basepath)


target_paths = joinpath.(basepath, target_names)
@assert all(ispath.(target_paths))

data_path = "data"

data_paths = joinpath.(target_paths, data_path)
@assert all(ispath.(target_paths))


function repr_sample_hist(column::T, nbins::Int, npoints::Int) where T<:AbstractVector
    n_per_bin = floor(Int, npoints/nbins)
    hist = fit(Histogram, column, nbins=nbins)
    bin_edges = hist.edges[1]

    idx_out = []

    # loop over each bin
    for i ∈ 1:size(bin_edges, 1)-1
        bin_idxs = findall(ξ -> bin_edges[i] < ξ && ξ < bin_edges[i+1], column)
        n_to_sample = minimum([n_per_bin, size(bin_idxs, 1)])
        idx_res = sample(bin_idxs, n_to_sample, replace=false)
        push!(idx_out, idx_res)  # sample without replacement
    end

    return unique(vcat(idx_out...))
end

function repr_sample_quantile(column::T, nbins::Int, npoints::Int) where T<:AbstractVector
    n_per_bin = floor(Int, npoints/nbins)
    bin_edges = quantile(column, range(0.0, stop=1.0, length=nbins))

    idx_out = []

    # loop over each bin
    for i ∈ 1:size(bin_edges, 1)-1
        bin_idxs = findall(ξ -> bin_edges[i] < ξ && ξ < bin_edges[i+1], column)
        n_to_sample = minimum([n_per_bin, size(bin_idxs, 1)])
        idx_res = sample(bin_idxs, n_to_sample, replace=false)
        push!(idx_out, idx_res)  # sample without replacement
    end

    return unique(vcat(idx_out...))
end



X = CSV.read(joinpath(data_paths[1], "X.csv"), DataFrame)
y = CSV.read(joinpath(data_paths[1], "y.csv"), DataFrame)


Features = names(X)

Features_repr = [
    "roll",
    "pitch",
    "heading",
    "view_angle",
    "solar_azimuth",
    "solar_elevation",
    "solar_zenith",
]

target = names(y)


cdom = @view y[!, 1]

idx_hist = repr_sample_hist(cdom, 10, 10000)
idx_quant = repr_sample_quantile(cdom, 10, 10000)



fig = Figure();
ax = Axis(fig[1,1], xlabel="CDOM", ylabel="counts");
h1 = hist!(ax, cdom[idx_hist], bins=100, color=(mints_colors[1], 0.6))
h2 = hist!(ax, cdom[idx_quant], bins=100, color=(mints_colors[2], 0.6))
ax = axislegend(ax, [h1, h2], ["histogram", "quantile"])
fig


# Based on this, I think we should go with the quantiles like Dr. Lary mentions in his paper.

fig = Figure();
ax = Axis(fig[1,1], xlabel="CDOM", ylabel="PDF")
d1 = density!(ax, cdom, label="full", color=(mints_colors[1],0.5), strokecolor=mints_colors[1], strokewidth=3)
d2 = density!(ax, cdom[idx_hist], label="histogram", color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=3)
d3 = density!(ax, cdom[idx_quant], label="quantile", color=(mints_colors[3], 0.5), strokecolor=mints_colors[3], strokewidth=3)
axislegend(ax)
fig


# so doing a quantile selection should help reduce impact of outliers by maintaining the original distribution while reducing total number of points...

# --> use 10,000 total points for an overall reduction in the data size... then we can do a 8000, 1000, 1000 split for train, cal , test split.



# let's do some further pre-analysis to identify any outliers in the data and drop them...
