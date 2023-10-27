# # https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
# ENV["GKSwstype"] = "100"

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
using Dates

include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")



# now that we've verified the cl args, move on with the script
println("Setting random seed for reproducability...")
rng = StableRNG(42)

println("Loading plotting theme...")
add_mints_theme()
theme(:mints)

println("Setting compute resources...")
# we should grab this from the envrionment variable for number of julia threads
MLJ.default_resource(CPUThreads())


# load in base models
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



# set up paths
modelbasepath = "/media/john/HSDATA/analysis_full"
basepath = "/media/john/HSDATA/processed"
dates = ["11-23", "12-09", "12-10"]
collections_11_23 = ["Scotty_1",
                     "Scotty_2",
                     "Scotty_3",
                     "Scotty_4",
                     "Scotty_5",
                     ]
collections_12_09 = ["NoDye_1",
                     "NoDye_2",
                     "Dye_1",
                     "Dye_2",
                     ]
collections_12_10 = ["NoDye_1",
                     "NoDye_2",
                     "Dye_1",
                     "Dye_2",
                     ]


paths_dict = Dict("11_23" => [joinpath(basepath, dates[1], c) for c ∈ collections_11_23],
                  "12_09" => [joinpath(basepath, dates[2], c) for c ∈ collections_12_09],
                  "12_10" => [joinpath(basepath, dates[3], c) for c ∈ collections_12_10],
                  )


function apply_models(date::String, mach, target)
    println("\t\t... generating file list")
    basepaths = paths_dict[date]
    fpaths = Array{String}(undef, 0)
    # generate list of files
    for basepath ∈ basepaths
        for f ∈ readdir(basepath)
            if endswith(f, ".csv") && !contains(f, "irradiance")
                push!(fpaths, joinpath(basepath, f))
            end
        end
    end

    for f ∈ fpaths
        # load the data
        X = CSV.File(f) |> DataFrame

        # SAVE WATER FOR MAPPING
        # filter to water only via modified normalized difference water index
        # X = X[X.mNDWI .> 0.25, :]

        # create utc_dt from utc_times
        if !("solar_az" ∈ names(X))
            date_format = DateFormat("yyyy-mm-dd H:M:S.s")
            X.utc_dt = DateTime.(X.utc_times, date_format)

            solar_geo = solar_azimuth_altitude.(X.utc_dt, X.ilat, X.ilon, X.altitude)
            az_el = hcat(collect.(solar_geo)...)'
            X.solar_az .= az_el[:, 1]
            X.solar_el .= az_el[:, 2]
        end

        # select relevant features
        others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]
        refs = [Symbol("λ_$(i)") for i ∈ 1:462]
        X̃ = X[:, vcat(refs, others)]

        # make predictions
        println("\t\t... applying model")
        pred = predict(mach, X̃);
        # update original dataframe and save
        X[!, target] = pred
        println("\t\t... saving output")
        CSV.write(f, X)
    end
end


for (target, info) ∈ targetsDict
    println("Working on $(target)")
    println("\tloading superlearner")
    modelpath = joinpath(modelbasepath, String(target), "superlearner_stack", "superlearner__stack.jls")
    ispath(modelpath);

    # load in trained SL
    mach = machine(modelpath)
    println("\tloading successful...")
    # apply to each dataset
    println("\tapplying to 11-23...")
    apply_models("11_23", mach, target)
    println("\tapplying to 12-09...")
    apply_models("12_09", mach, target)
    println("\tapplying to 12-10...")
    apply_models("12_10", mach, target)
end

