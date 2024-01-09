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
    "11-23" => Dict(
        "Scotty_1" => "/Users/johnwaczak/data/robot-team/finalized/11-23/Scotty_1",
        "Scotty_2" => "/Users/johnwaczak/data/robot-team/finalized/11-23/Scotty_2",
    ),
    "12-09" => Dict(
        "NoDye_1" => "/Users/johnwaczak/data/robot-team/finalized/12-09/NoDye_1",
        "NoDye_2" => "/Users/johnwaczak/data/robot-team/finalized/12-09/NoDye_2",
    ),
    "12-10" => Dict(
        "NoDye_1" => "/Users/johnwaczak/data/robot-team/finalized/12-10/NoDye_1",
        "NoDye_2" => "/Users/johnwaczak/data/robot-team/finalized/12-10/NoDye_2",
    ),
)


dfs = Dict(
    "11-23" => Dict(
        "Scotty_1" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
        "Scotty_2" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
    ),
    "12-09" => Dict(
        "NoDye_1" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
        "NoDye_2" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
    ),
    "12-10" => Dict(
        "NoDye_1" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
        "NoDye_2" => Dict(
            "df" => DataFrame(),
            "feature_names" => [],
            "target_names" => [],
        ),
    ),
)


# inpath = "/Volumes/LabData/RobotTeam/finalized/Full"
for (collection, flights) in inpaths
    for (flight, path) in flights
        @assert ispath(path)
    end
end


nan_cols = String[]
inf_cols = String[]

# read in the datasets:
function get_data(collection, flight; threshold=0.3)
    dfs = DataFrame[]
    feature_names = []
    target_names = []


    # get unique list of file basenames

    fnames = unique([split(n, "__")[1] for n in readdir(inpaths[collection][flight])])
    bad_names = unique([split(n, ".")[1] for n in bad_hsi_dict[collection][flight]])
    fnames = [n for n in fnames if !(n in bad_names)]

    for fname in fnames
        let
            df_features = CSV.read(joinpath(inpaths[collection][flight], fname*"__Features.csv"), DataFrame)
            df_targets = CSV.read(joinpath(inpaths[collection][flight], fname*"__Targets.csv"), DataFrame)

            if isempty(feature_names)
                push!(feature_names, names(df_features)...)
            end

            if isempty(target_names)
                push!(target_names, names(df_targets)...)
            end

            df = hcat(df_features, df_targets)
            push!(dfs, df)
        end
    end

    @info "Joining tables together"
    df = vcat(dfs...)

    # first select only those points that are identifiably "in" the water and deal with any rows with missing data
    filter!(row -> row.NDWI1 > threshold, df);
    dropmissing!(df);

    # identify nan/inf columns
    for i ∈ 1:ncol(df)
        try
            if any(isnan.(df[:,i]))
                println("Nan in $(names(df)[i])")
                if !(names(df)[i] in nan_cols)
                    push!(nan_cols, names(df)[i])
                end
            end
        catch
            println("\t $(names(df)[i]) not numeric")
        end
    end

    for i ∈ 1:ncol(df)
        try
            if any(isinf.(df[:,i]))
                println("Inf in $(names(df)[i])")
                if !(names(df)[i] in inf_cols)
                    push!(inf_cols, names(df)[i])
                end
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
        "unix_dt",
        "utc_dt",
        "zone",
        "isnorth",
        "X",
        "Y",
        "category",
        "predye_postdye",
    ]
    # ignorecols = vcat(ignorecols, nan_cols, inf_cols)


    data = data[:, Not(ignorecols)]

    target_names = [t for t ∈ target_names if !(t∈ignorecols)]
    feature_names = [f for f ∈ feature_names if !(f∈ignorecols)]

    return data, feature_names, target_names
end




# fill in the dataframes list
for (collection, flights) in inpaths
    for (flight, _) in flights
        data, feature_names, target_names = get_data(collection, flight);

        dfs[collection][flight]["df"] = data
        dfs[collection][flight]["feature_names"] = feature_names
        dfs[collection][flight]["target_names"] = target_names
    end
end


# remove the bad columns
for (collection, flights) in inpaths
    for (flight, _) in flights
        dfs[collection][flight]["df"] = dfs[collection][flight]["df"][:, Not([nan_cols..., inf_cols...])]
    end
end



function make_datasets(df, target, feature_names)
    df_use = df[:, [feature_names..., target]]

    y, X = unpack(df_use, ==(Symbol(target)), col -> col ∈ Symbol.(feature_names))

    if length(targets_dict[Symbol(target)]) == 3
        ymin = targets_dict[Symbol(target)][3]
        y[y .< ymin] .= 0.0
    end

    return (y, X)
end



# collect background satellite map for distribution plots
w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413
satmap = get_background_satmap(w,e,s,n)




df_11_23 = vcat(
    dfs["11-23"]["Scotty_1"]["df"],
    dfs["11-23"]["Scotty_2"]["df"],
)

df_12_09 = vcat(
    dfs["12-09"]["NoDye_1"]["df"],
    dfs["12-09"]["NoDye_2"]["df"],
)

df_12_10 = vcat(
    dfs["12-10"]["NoDye_1"]["df"],
    dfs["12-10"]["NoDye_2"]["df"],
)


df_full = vcat(
    dfs["11-23"]["Scotty_1"]["df"],
    dfs["11-23"]["Scotty_2"]["df"],
    dfs["12-09"]["NoDye_1"]["df"],
    dfs["12-09"]["NoDye_2"]["df"],
    dfs["12-10"]["NoDye_1"]["df"],
    dfs["12-10"]["NoDye_2"]["df"],
)


target_names = dfs["11-23"]["Scotty_1"]["target_names"]

fnames_11_23 = names(df_11_23[:, Not(target_names)])
fnames_12_09 = names(df_12_09[:, Not(target_names)])
fnames_12_10 = names(df_12_10[:, Not(target_names)])
fnames_full = names(df_full[:, Not(target_names)])


df_latlong_11_23 = DataFrame(
    :longitudes => df_11_23.longitude,
    :latitudes => df_11_23.latitude,
)
df_latlong_12_09 = DataFrame(
    :longitudes => df_12_09.longitude,
    :latitudes => df_12_09.latitude,
)
df_latlong_12_10= DataFrame(
    :longitudes => df_12_10.longitude,
    :latitudes => df_12_10.latitude,
)
df_latlong_full = DataFrame(
    :longitudes => df_full.longitude,
    :latitudes => df_full.latitude,
)


df_11_23 = df_11_23[:, Not([:longitude, :latitude])]
df_12_09 = df_12_09[:, Not([:longitude, :latitude])]
df_12_10 = df_12_10[:, Not([:longitude, :latitude])]
df_full = df_full[:, Not([:longitude, :latitude])]



collections = ["11-23", "12-09", "12-10", "Full"]
dfs_out = [df_11_23, df_12_09, df_12_10, df_full]
dfs_latlong = [df_latlong_11_23, df_latlong_12_09, df_latlong_12_10, df_latlong_full]
feature_names_all = [fnames_11_23, fnames_12_09, fnames_12_10, fnames_full]


for i in 1:length(collections)
    collection = collections[i]
    df_out = dfs_out[i]
    df_latlong = dfs_latlong[i]
    feature_names = feature_names_all[i]

    @info "Working on $(collection)"


    basepath = joinpath(outpath, collection)
    if !ispath(basepath)
        mkpath(basepath)
    end

    maps_path = joinpath(mapspath, collection)
    if !ispath(maps_path)
        mkpath(maps_path)
    end


    @showprogress for (target, info) in targets_dict
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

        (y,X) = make_datasets(df_out, target_name, feature_names)
        idx_train, idx_test = partition(eachindex(y), 0.9, shuffle=true, rng=rng)


        CSV.write(joinpath(outpath_final, "idx_train.csv"), DataFrame(Dict(:idx => idx_train)))
        CSV.write(joinpath(outpath_final, "idx_test.csv"), DataFrame(Dict(:idx => idx_test)))

        CSV.write(joinpath(outpath_final, "X.csv"), X)
        CSV.write(joinpath(outpath_final, "y.csv"), DataFrame(Dict(target => y)))
        CSV.write(joinpath(outpath_final, "lat_lon.csv"), df_latlong)

    end
end


