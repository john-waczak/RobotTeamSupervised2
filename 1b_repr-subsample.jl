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


# other utility packages
using ProgressMeter

# for partition
using MLJ: partition, unpack

using Random

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


include("./config.jl")


basepath = "/media/teamlary/LabData/RobotTeam/supervised"

@assert ispath(basepath)

target_names = String.(keys(targets_dict))
target_paths = joinpath.(basepath, target_names)
@assert all(ispath.(target_paths))


data_path = "data"
save_path = "data-repr"
data_paths = joinpath.(target_paths, data_path)
@assert all(ispath.(target_paths))


function repr_sample_quantile(column::T, nbins::Int, npoints::Int) where T<:AbstractVector
    n_per_bin = floor(Int, npoints/nbins)
    bin_edges = quantile(column, range(0.0, stop=1.0, length=nbins+1))

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




# Try this out for the first one:

X = DataFrame()
y = DataFrame()

ntrain = 0
ntest = 0

let
    Xtrain = CSV.read(joinpath(data_paths[28], "X.csv"), DataFrame)
    Xtest = CSV.read(joinpath(data_paths[28], "Xtest.csv"), DataFrame)

    ytrain = CSV.read(joinpath(data_paths[28], "y.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_paths[28], "ytest.csv"), DataFrame)


    global ntrain = nrow(Xtrain)
    global ntest = nrow(Xtest)

    global X = vcat(Xtrain, Xtest)
    global y = vcat(ytrain, ytest)
end

ntot = nrow(X)

@assert ntot == ntrain + ntest

println((10_000 / ntot)*100)



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

idx_quant = repr_sample_quantile(cdom, 10, 10_000)


fig = Figure();
ax = Axis(fig[1,1], xlabel="CDOM", ylabel="PDF")
d1 = density!(ax, cdom, label="full", color=(mints_colors[1],0.5), strokecolor=mints_colors[1], strokewidth=3)
d2 = density!(ax, cdom[idx_quant], label="quantile", color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=3)
fig


idx_traincal, idx_test = partition(1:10_000, 0.9, shuffle=true, rng=rng)

idx_1 = idx_quant[idx_traincal]
idx_2 = idx_quant[idx_test]


d3 = density!(ax, cdom[idx_1], label="quantile train", color=(mints_colors[3], 0.5), strokecolor=mints_colors[3], strokewidth=3)
d4 = density!(ax, cdom[idx_2], label="quantile test", color=(mints_colors[4], 0.5), strokecolor=mints_colors[4], strokewidth=3)
axislegend(ax)
fig
# so now we can take the indices and create smaller datasets...

# --> use 10,000 total points for an overall reduction in the data size... then we can do a 8000, 1000, 1000 split for train, cal , test split.


@showprogress for (target, info) in targets_dict
    target_name = String(target)
    target_name = String(target)

    inpath = joinpath(basepath, target_name, "data")

    outpath = joinpath(basepath, target_name, "data-repr")
    if !ispath(outpath)
        mkpath(outpath)
    end

    X = DataFrame()
    y = DataFrame()

    ntrain = 0
    ntest = 0
    let
        Xtrain = CSV.read(joinpath(inpath, "X.csv"), DataFrame)
        Xtest = CSV.read(joinpath(inpath, "Xtest.csv"), DataFrame)

        ytrain = CSV.read(joinpath(inpath, "y.csv"), DataFrame)
        ytest = CSV.read(joinpath(inpath, "ytest.csv"), DataFrame)

        global ntrain = nrow(Xtrain)
        global ntest = nrow(Xtest)

        global X = vcat(Xtrain, Xtest)
        global y = vcat(ytrain, ytest)
    end

    ntot = nrow(X)

    @assert ntot == ntrain + ntest


    # get representative sample of
    idx_quant = repr_sample_quantile(cdom, 10, 10_000)
    idx_1, idx_2= partition(1:10_000, 0.9, shuffle=true, rng=rng)

    idx_train = idx_quant[idx_1]
    idx_test = idx_quant[idx_2]

    # save the data to the outpaths
    CSV.write(joinpath(outpath, "X.csv"), X[idx_train, :])
    CSV.write(joinpath(outpath, "y.csv"), y[idx_train,:])
    CSV.write(joinpath(outpath, "Xtest.csv"), X[idx_test, :])
    CSV.write(joinpath(outpath, "ytest.csv"), y[idx_test, :])
end



# add another folder with data converted to float32

# add another folder with representative subsample

