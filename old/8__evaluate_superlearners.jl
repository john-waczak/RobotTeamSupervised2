# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"


using Pkg
Pkg.instantiate()

using mintsML
using MLJ
using Flux, MLJFlux
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
            default = "/media/john/HSDATA/datasets/11-23"
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


function main()
    # parse args making sure that supplied target does exist
    parsed_args = parse_commandline()
    target = parsed_args[:target]
    datapath = parsed_args[:datapath]
    outpath = parsed_args[:outpath]

    date = split(datapath, "/")[end]

    target_name = String(target)
    target_long = targetsDict[target][2]
    units = targetsDict[target][1]

    # now that we've verified the cl args, move on with the script
    println("Setting random seed for reproducability...")
    rng = StableRNG(42)

    println("Loading plotting theme...")
    add_mints_theme()
    theme(:mints)

    println("Setting compute resources...")
    # we should grab this from the envrionment variable for number of julia threads
    MLJ.default_resource(CPUThreads())

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



    # set up reading and outgoing paths
    longname = "Superlearner Stack"
    savename = "superlearner"
    suffix = "stack"


    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpath_stack = joinpath(outpathtarget, "superlearner_stack")
    if !isdir(outpath_stack)
        mkdir(outpath_stack)
    end


    path_to_use = outpath_stack
    mpath = joinpath(path_to_use, "$(savename)__$(suffix).jls")
    mach = machine(mpath)

    ŷ = MLJ.predict(mach, X̃)
    ŷtest = MLJ.predict(mach, X̃test)

    p1 = scatterresult(y, ŷ,
                       ytest, ŷtest;
                       xlabel="True $(target_long) [$(units)]",
                       ylabel="Predicted $(target_long) [$(units)]",
                       plot_title="Fit for $(longname)",)

    savefig(p1, joinpath(path_to_use, "scatterplt__$(date).png"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(date).svg"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(date).pdf"))

    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          xlabel="True $(target_long) [$(units)]",
                          ylabel="Predicted $(target_long) [$(units)]",
                          title="Fit for $(longname)",)

    savefig(p2, joinpath(path_to_use, "qq__$(date).png"))
    savefig(p2, joinpath(path_to_use, "qq__$(date).svg"))
    savefig(p2, joinpath(path_to_use, "qq__$(date).pdf"))

    open(joinpath(path_to_use, "$(savename)__$(date).txt"), "w") do f
        println(f,"\n")
        println(f,"---------------------")
        println(f, "r² train: $(rsq(ŷ, y))\tr² test:$(rsq(ŷtest, ytest))\tRMSE test: $(rmse(ŷtest, ytest))\tMAE test: $(mae(ŷtest, ytest))")
    end
end


DTR = @load DecisionTreeRegressor pkg=DecisionTree
RFR = @load RandomForestRegressor pkg=DecisionTree
XGBR = @load XGBoostRegressor pkg=XGBoost
KNNR = @load KNNRegressor pkg=NearestNeighborModels
ETR = @load EvoTreeRegressor pkg=EvoTrees
LGBR = @load LGBMRegressor pkg=LightGBM
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
LR = @load LassoRegressor pkg=MLJLinearModels
LinearR = @load LinearRegressor pkg=MLJLinearModels
RR = @load RidgeRegressor pkg=MLJLinearModels

main()
