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


using MLJ, Flux, ConformalPrediction

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


# pull in targets info
include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")

datapath = "/media/teamlary/LabData/RobotTeam/supervised"


RFR = @load RandomForestRegressor pkg=DecisionTree
XGBR = @load XGBoostRegressor pkg=XGBoost
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels



# 1. Construct dictionary of models
MODELS = Dict()


# the neural network needs features to be standardized
nnr = NNR(builder=MLJFlux.MLP(hidden=(1000,), σ=NNlib.relu),
          batch_size = 200,
          optimiser=Flux.Optimise.ADAM(0.001),
          lambda = 0.0001,  # default regularization strength (I'm a bit confused by this as sk-learn only has alpha but that seems different here)
          rng=42,
          epochs=200,
          )

MODELS[:nnr] = (;
                :longname=>"Neural Network Regressor",
                :savename => "NeuralNetworkRegressor",
                :mdl => Standardizer() |> nnr
                )

# 3. Add XGBoostRegressor. Defaults seem fine...
MODELS[:xgbr] = (;
                 :longname=>"XGBoost Regressor",
                 :savename=>"XGBoostRegressor",
                 :mdl => XGBR()
                 )

# 4. Add Random Forest using sk-learn defaults
rfr = RFR(;
          n_trees = 100,
          max_depth = -1,
          min_samples_split=2,
          min_samples_leaf=1,
          min_purity_increase=0,
          n_subfeatures=0,
          sampling_fraction=1.0,
          )
MODELS[:rfr] = (;
                :longname=>"Random Forest Regressor",
                :savename=>"RandomForestRegressor",
                :mdl=>rfr
                )



# 5. Fit each of the models to different subsets of features.
targets_to_try = [:CDOM, :CO, :Na, :Cl]


for target ∈ targets_to_try
    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]


    @info "Working on $(target_name)"


    data_path = joinpath(datapath, target_name, "data")
    outpath = joinpath(datapath, target_name, "models")
    if !ispath(outpath)
        mkpath(outpath)
    en

    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]

    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]


    results_summary = []

    for (shortname, model) ∈ MODELS
        try
            res = train_basic(X, y,
                              Xtest, ytest,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              )

            push!(results_summary, res)
        catch e
            println("\t$(e)")
        end
    end

    println("\tsaving results")

    res_df = DataFrame(results_summary, [:rsq_train, :rsq_test, :rmse_train, :rmse_test, :emp_cov])
    CSV.write(joinpath(outpath, "$(target_name)_model_comparison.csv"), res_df)
end

