# data tools
using CSV, DataFrames

# other utility packages
using ProgressMeter

# for partition
using MLJ: partition, unpack

using Random

import CairoMakie as cmk
using MintsMakieRecipes

cmk.set_theme!(mints_theme)
cmk.update_theme!(
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


using StatsBase, Statistics

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)

# include our config file with targets dict
include("./config.jl")
include("./viz.jl")
include("./sat-viz.jl")


outpath = "/Users/johnwaczak/data/robot-team/supervised"
if !ispath(outpath)
    mkpath(outpath)
end

mapspath = "/Users/johnwaczak/data/robot-team/maps"
if !ispath(mapspath)
    mkpath(mapspath)
end


inpaths = Dict(
    "11-23" => "/Users/johnwaczak/data/robot-team/finalized/11-23",
    "Full" => "/Users/johnwaczak/data/robot-team/finalized/Full",
    "12-09" => "/Users/johnwaczak/data/robot-team/finalized/12-09",
    "12-10" => "/Users/johnwaczak/data/robot-team/finalized/12-10",
)

# inpath = "/Volumes/LabData/RobotTeam/finalized/Full"
for (collection, path) in inpaths
    @assert ispath(path)
end


# collect background satellite map for distribution plots
w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413
satmap = get_background_satmap(w,e,s,n)


# df = CSV.read("/Users/johnwaczak/data/robot-team/prepared/12-10/Targets.csv", DataFrame)

# gdf = groupby(df, :category)

# fig = cmk.Figure();
# ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
# bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# for (cat, df_sub) in pairs(gdf)
#     s = cmk.scatter!(ax, df_sub.longitude, df_sub.latitude, alpha=0.75, label=cat.category)
# end
# cmk.axislegend(ax)
# fig

# df_preflight = gdf[(category="Dye_1_preflight",)]
# df_preflight = df_preflight[df_preflight.predye_postdye .== "Pre-Dye", :]

# fig = cmk.Figure();
# ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
# bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# s = cmk.scatter!(ax, df_preflight.longitude[1:end-800], df_preflight.latitude[1:end-800], color=df_preflight.CDOM[1:end-800], markersize=3)
# cb = cmk.Colorbar(fig[1,2], s)
# fig

# gdf = groupby(df, :predye_postdye)
# df_predye = gdf[(predye_postdye="Pre-Dye",)]

# fig = cmk.Figure();
# ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
# bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# s = cmk.scatter!(ax, df_predye.longitude[:], df_predye.latitude[:], color=df_predye.CDOM[:], markersize=3)
# cb = cmk.Colorbar(fig[1,2], s)
# fig




# read in the datasets:
function get_data(fpath; threshold=0.25)
    df = DataFrame()
    feature_names = []
    target_names = []
    let
        # df_features = vcat(CSV.read(joinpath(fpath, "Features_1.csv"), DataFrame), CSV.read(joinpath(fpath, "Features_2.csv"), DataFrame))
        # df_targets = vcat(CSV.read(joinpath(fpath, "Targets_1.csv"), DataFrame), CSV.read(joinpath(fpath, "Targets_2.csv"), DataFrame))

        df_features = CSV.read(joinpath(fpath, "Features_2.csv"), DataFrame)
        df_targets = CSV.read(joinpath(fpath, "Targets_2.csv"), DataFrame)

        # df_features = CSV.read(joinpath(fpath, "Features_1.csv"), DataFrame)
        # df_targets = CSV.read(joinpath(fpath, "Targets_1.csv"), DataFrame)



        feature_names = names(df_features)
        target_names = names(df_targets)

        df = hcat(df_features, df_targets)
    end

    # first select only those points that are identifiably "in" the water and deal with any rows with missing data
    filter!(row -> row.mNDWI > threshold, df);
    dropmissing!(df);

    # identify nan/inf columns
    nan_cols = String[]
    for i ∈ 1:ncol(df)
        try
            if any(isnan.(df[:,i]))
                println("Nan in $(names(df)[i])")
                push!(nan_cols, names(df)[i])
            end
        catch
            println("\t $(names(df)[i]) not numeric")
        end
    end

    inf_cols = String[]
    for i ∈ 1:ncol(df)
        try
            if any(isinf.(df[:,i]))
                println("Inf in $(names(df)[i])")
                push!(inf_cols, names(df)[i])
            end
        catch
            println("\t $(names(df)[i]) not numeric")
        end
    end

    # make sure we only keep the pre-dye data
    gdf = groupby(df, :predye_postdye)
    data = gdf[(predye_postdye="Pre-Dye",)]

    longitudes = data.longitude
    latitudes = data.latitude


    ignorecols = [
        "latitude",
        "longitude",
        "unix_dt",
        "utc_dt",
        "zone",
        "isnorth",
        "X",
        "Y",
        "category",
        "predye_postdye",
    ]
    ignorecols = vcat(ignorecols, nan_cols, inf_cols)


    data = data[:, Not(ignorecols)]
    target_names = [t for t ∈ target_names if !(t∈ignorecols)]
    feature_names = [f for f ∈ feature_names if !(f∈ignorecols)]

    return data, feature_names, target_names, ignorecols, longitudes, latitudes
end


function make_datasets(data, target, feature_names, target_names, ignorecols)
    df_use = data[:, [feature_names..., target]]

    y, X = unpack(df_use, ==(Symbol(target)), col -> !(col∈[target_names..., ignorecols...]))

    if length(targets_dict[Symbol(target)]) == 3
        ymin = targets_dict[Symbol(target)][3]
        y[y .< ymin] .= 0.0
    end

    return (y, X)
end


# data, feature_names, target_names, ignorecols = get_data(inpaths["11-23"])
# (y,X) = make_datasets(data, "CDOM", feature_names, target_names, ignorecols)


for (collection, fpath) in inpaths
    basepath = joinpath(outpath, collection)
    if !ispath(basepath)
        mkpath(basepath)
    end

    maps_path = joinpath(mapspath, collection)
    if !ispath(maps_path)
        mkpath(maps_path)
    end


    data, feature_names, target_names, ignorecols, longitudes, latitudes = get_data(fpath)

    df_latlon = DataFrame(
        :longitudes => longitudes,
        :latitudes => latitudes
    )

    @showprogress for (target, info) ∈ targets_dict
        target_name = String(target)
        target_long = targets_dict[target][2]
        units = targets_dict[target][1]


        outpath_final = joinpath(basepath, target_name, "data")
        if !ispath(outpath_final)
            mkpath(outpath_final)
        end

        mapspath_final = joinpath(maps_path, target_name)
        if !ispath(mapspath_final)
            mkpath(mapspath_final)
        end

        fig = cmk.Figure();
        ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
        bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
        sc = cmk.scatter!(ax, longitudes, latitudes, color=data[:, target], markersize=3)
        cb = cmk.Colorbar(fig[1,2], sc, label="$(target_long) ($(units))")

        save(joinpath(mapspath_final, "data-map.png"), fig)
        save(joinpath(mapspath_final, "data-map.pdf"), fig)

        (y,X) = make_datasets(data, target_name, feature_names, target_names, ignorecols)
        idx_train, idx_test = partition(eachindex(y), 0.9, shuffle=true, rng=rng)


        CSV.write(joinpath(outpath_final, "idx_train.csv"), DataFrame(Dict(:idx => idx_train)))
        CSV.write(joinpath(outpath_final, "idx_test.csv"), DataFrame(Dict(:idx => idx_test)))

        CSV.write(joinpath(outpath_final, "X.csv"), X)
        CSV.write(joinpath(outpath_final, "y.csv"), DataFrame(Dict(target => y)))
        CSV.write(joinpath(outpath_final, "lat_lon.csv"), df_latlon)
    end
end
