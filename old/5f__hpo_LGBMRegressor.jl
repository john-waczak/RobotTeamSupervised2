# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"


using Pkg
Pkg.instantiate()

using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ProgressMeter
using StableRNGs
using ArgParse

include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--target","-T"
            help = "Target variable for training. See targetsDict in config.jl for full list."
            arg_type = Symbol
            default = :CDOM
        "--datapath", "-d"
            help = "Path to directory with data to be used in training"
            arg_type = String
            default = "/media/john/HSDATA/datasets/Full"
        "--outpath", "-o"
            help = "Path to directory for storing output"
            arg_type = String
            default ="/media/john/HSDATA/analysis_full"
    end


    parsed_args = parse_args(ARGS, s; as_symbols=true)

    @assert (parsed_args[:target] ∈ keys(targetsDict)) "Supplied target does not exist"

    # make sure that the datapath and outpath exist
    if !isdir(parsed_args[:datapath])
        println("$(parsed_args[:datapath]) does not exist. Creating now...")
        mkpath(parsed_args[:datapath])
    end

    if !isdir(parsed_args[:outpath])
        println("$(parsed_args[:outpath]) does not exist. Creating now...")
        mkpath(parsed_args[:outpath])
    end

    return parsed_args
end


function main(mdl)
    # parse args making sure that supplied target does exist
    parsed_args = parse_commandline()
    target = parsed_args[:target]
    datapath = parsed_args[:datapath]
    outpath = parsed_args[:outpath]

    target_name = String(target)
    target_long = targetsDict[target][2]
    units = targetsDict[target][1]

    # now that we've verified the cl args, move on with the script
    println("Loading packages...")

    println("Setting random seed for reproducability...")
    rng = StableRNG(42)

    println("Loading plotting theme...")
    add_mints_theme()
    theme(:mints)

    println("Setting compute resources...")
    # we should grab this from the envrionment variable for number of julia threads
    # MLJ.default_resource(CPUThreads())


    println("Loading datasets...")
    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    println("Reducing to reflectance + geometry feature set...")
    others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]
    refs = [Symbol("λ_$(i)") for i ∈ 1:462]
    X̃ = X[:, vcat(refs, others)]
    X̃test = Xtest[:, vcat(refs, others)]


    train_hpo(y, X̃,
              ytest, X̃test,
              "Light GBM Regressor", "LGBMRegressor", "LightGBM", mdl,
              target_name, units, target_long,
              outpath;
              nmodels = 100,
              accelerate=false
              )



end


model = @load LGBMRegressor pkg=LightGBM
mdl = model()
main(mdl)
