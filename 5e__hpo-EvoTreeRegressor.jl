# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"

using Pkg
Pkg.activate(".")
Pkg.instantiate()



using CSV, DataFrames
using ProgressMeter
using Statistics
using Random

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


#using MLJ, Flux, ConformalPrediction
using MLJ
using ArgParse

Pkg.add(url="https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl.git")
using ConformalPrediction



# pull in targets info
include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--target_index","-i"
            help = "Index of target variable for training. See targets_dict in config.jl for full list."
            arg_type = Int
            default = 1
        "--datapath", "-d"
            help = "Path to directory with data to be used in training"
            arg_type = String
            default = "/media/teamlary/LabData/RobotTeam/supervised"
    end


    parsed_args = parse_args(ARGS, s; as_symbols=true)

    idx_target = parsed_args[:target_index]
    @assert idx_target  > 0 && idx_target <= length(keys(targets_dict))
    @assert ispath(parsed_args[:datapath]) "datapath does not exist"

    return parsed_args
end




function main(mdl)
    # seed reproducible pseudo-random number generator
    @info "Setting random number seed"
    rng = Xoshiro(42)

    # parse args making sure that supplied target does exist
    parsed_args = parse_commandline()

    idx_target = parsed_args[:target_index]
    target_keys = [k for k in keys(targets_dict)]
    target = target_keys[idx_target]

    datapath = parsed_args[:datapath]


    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    data_path = joinpath(datapath, target_name, "data")
    @assert ispath(data_path)
    outpath = joinpath(datapath, target_name, "models")
    @assert ispath(outpath)


    @info "Setting default compute resource to CPU..."
    # we should grab this from the envrionment variable for number of julia threads
    MLJ.default_resource(CPUThreads())


    @info "Loading datasets..."
    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]


    train_hpo(
        X, y,
        Xtest, ytest,
        "EvoTree Regressor", "EvoTreeRegressor", "EvoTrees",
        target_name, units, target_long,
        mdl,
        outpath;
    )
end


model = @load EvoTreeRegressor pkg=EvoTrees
mdl = model()
main(mdl)
