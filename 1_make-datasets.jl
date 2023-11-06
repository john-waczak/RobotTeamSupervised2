# data tools
using CSV, DataFrames

# other utility packages
using ProgressMeter

# for partition
using MLJ: partition, unpack

using Random

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)

# include our config file with targets dict
include("./config.jl")


outpath = "/media/teamlary/LabData/RobotTeam/supervised"
if !ispath(outpath)
    mkpath(outpath)
end

inpath = "/media/teamlary/LabData/RobotTeam/finalized/Full"
@assert ispath(inpath)



# load features and targets into single dataframe
df = DataFrame[];
feature_names = String[];
target_names = String[];

let
    df_features = CSV.read(joinpath(inpath, "Features.csv"), DataFrame)
    for name ∈ names(df_features)
        push!(feature_names, name)
    end
    push!(df, df_features)

    df_targets = CSV.read(joinpath(inpath, "Targets.csv"), DataFrame)
    for name ∈ names(df_targets)
        push!(target_names, name)
    end
    push!(df, df_targets)
end

df = hcat(df...)

# remove any rows with missing
dropmissing!(df)

# identify missing columns
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


function make_datasets(data, target; split_fracs=[0.8, 0.1], feature_names=feature_names, target_names=target_names, rng=rng)

    df_use = data[:, [feature_names..., target]]
    dropmissing!(df_use)

    df, df_test, df_cal = partition(df_use, split_fracs...; rng)

    # now split into (X,y) feature-target pairs
    y, X = unpack(df, ==(Symbol(target)), col -> !(col∈[target_names..., ignorecols...]))
    ytest, Xtest = unpack(df_test, ==(Symbol(target)), col -> !(col∈[target_names..., ignorecols...]))
    ycal, Xcal = unpack(df_cal, ==(Symbol(target)), col -> !(col∈[target_names..., ignorecols...]))

    # if there's a third value in targets_dict, set everything below the min to zero
    if length(targets_dict[Symbol(target)]) == 3
        ymin = targets_dict[Symbol(target)][3]
        y[y .< ymin] .= 0.0
        ytest[ytest .< ymin] .= 0.0
        ycal[ycal .< ymin] .= 0.0
    end

    return (y, X), (ytest, Xtest), (ycal, Xcal)
end


@showprogress for (target, info) ∈ targets_dict
    target_name = String(target)
    outpath_base = joinpath(outpath, target_name, "data")
    if !ispath(outpath_base)
        mkpath(outpath_base)
    end

    (y,X), (ytest,Xtest), (ycal, Xcal) = make_datasets(data, target_name);

    # save the data to the outpaths
    CSV.write(joinpath(outpath_base, "X.csv"), X)
    CSV.write(joinpath(outpath_base, "y.csv"), DataFrame(Dict(target => y)))

    CSV.write(joinpath(outpath_base, "Xtest.csv"), Xtest)
    CSV.write(joinpath(outpath_base, "ytest.csv"), DataFrame(Dict(target => ytest)))

    CSV.write(joinpath(outpath_base, "Xcal.csv"), Xcal)
    CSV.write(joinpath(outpath_base, "ycal.csv"), DataFrame(Dict(target => ycal)))
end


