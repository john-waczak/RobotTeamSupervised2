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


# for base model prediction:
# using MLJModelInterface: reformat
# predict(conf_model.model, mach.fitresult, reformat(conf_model.model, Xtest)...)

# we may want to convert all the data to Float32 first for faster use in the Neural Network.

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


# pull in targets info
include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")

datapath = "/media/teamlary/LabData/RobotTeam/supervised"

# DecisionTreeRegressor and RandomForestRegressor don't seem to work.
# RFR = @load RandomForestRegressor pkg=DecisionTree
# DTR = @load DecisionTreeRegressor pkg=DecisionTree

# julia> fit!(mach)
# [ Info: Training machine(SimpleInductiveRegressor(model = DecisionTreeRegressor(max_depth = -1, …), …), …).
# ┌ Error: Problem fitting the machine machine(SimpleInductiveRegressor(model = DecisionTreeRegressor(max_depth = -1, …), …), …). 
# └ @ MLJBase ~/.julia/packages/MLJBase/ByFwA/src/machines.jl:682
# [ Info: Running type checks... 
# [ Info: Type checks okay. 
# ERROR: MethodError: no method matching fit(::MLJDecisionTreeInterface.DecisionTreeRegressor, ::Int64, ::Matrix{Float64}, ::Vector{Float64})

# Closest candidates are:
#   fit(::MLJDecisionTreeInterface.DecisionTreeRegressor, ::Int64, ::Any, ::Any, ::Any)                                                                          
#    @ MLJDecisionTreeInterface ~/.julia/packages/MLJDecisionTreeInterface/cIWGa/src/MLJDecisionTreeInterface.jl:284
#   fit(::Supervised, ::Any, ::Any, ::Any, ::Any)
#    @ MLJModelInterface ~/.julia/packages/MLJModelInterface/io0Lg/src/model_api.jl:16                                                                 
#   fit(::NetworkComposite, ::Any, ::Any...)
#    @ MLJBase ~/.julia/packages/MLJBase/ByFwA/src/composition/models/network_composite.jl:23                                                         
#   ...

# Stacktrace:
#  [1] fit(conf_model::ConformalPrediction.SimpleInductiveRegressor{MLJDecisionTreeInterface.DecisionTreeRegressor}, verbosity::Int64, X::DataFrame, y::Vector{Float64})                                                                      
#    @ ConformalPrediction ~/.julia/packages/ConformalPrediction/z9UQ8/src/conformal_models/inductive_regression.jl:42
#  [2] fit_only!(mach::Machine{ConformalPrediction.SimpleInductiveRegressor{MLJDecisionTreeInterface.DecisionTreeRegressor}, true}; rows::Nothing, verbosity::Int64, force::Bool, composite::Nothing)                                         
#    @ MLJBase ~/.julia/packages/MLJBase/ByFwA/src/machines.jl:680
#  [3] fit_only!
#    @ ~/.julia/packages/MLJBase/ByFwA/src/machines.jl:606 [inlined]
#  [4] #fit!#63
#    @ ~/.julia/packages/MLJBase/ByFwA/src/machines.jl:778 [inlined]
#  [5] fit!(mach::Machine{ConformalPrediction.SimpleInductiveRegressor{MLJDecisionTreeInterface.DecisionTreeRegressor}, true})
#    @ MLJBase ~/.julia/packages/MLJBase/ByFwA/src/machines.jl:775
#  [6] top-level scope
#    @ REPL[55]:1




XGBR = @load XGBoostRegressor pkg=XGBoost
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels
ETR = @load EvoTreeRegressor pkg=EvoTrees


# 1. Construct dictionary of models
MODELS = Dict()

ref_cols = Symbol.(["R_" * lpad(i, 3, "0") for i ∈ 1:length(wavelengths)])


# the neural network needs features to be standardized
nnr = NNR(builder=MLJFlux.MLP(hidden=(1000,), σ=Flux.relu),
          batch_size = 200,
          optimiser=Flux.Optimise.ADAM(0.001),
          lambda = 0.0001,  # default regularization strength (I'm a bit confused by this as sk-learn only has alpha but that seems different here)
          rng=42,
          epochs=200,
          )

MODELS[:nnr] = (;
                :longname=>"Neural Network Regressor",
                :savename => "NeuralNetworkRegressor",
                #:mdl => Standardizer(features = name -> !(name ∈ ref_cols) ) |> nnr
                :mdl => nnr
                )


# 3. Add XGBoostRegressor. Defaults seem fine...
MODELS[:xgbr] = (;
                 :longname=>"XGBoost Regressor",
                 :savename=>"XGBoostRegressor",
                 :mdl => XGBR()
                 )


MODELS[:etr] = (;
                :longname=>"EvoTree Regressor",
                :savename=>"EvoTreeRegressor",
                :mdl => ETR()
                )



# 4. Add Random Forest using sk-learn defaults
# rfr = RFR(;
#           n_trees = 100,
#           max_depth = -1,
#           min_samples_split=2,
#           min_samples_leaf=1,
#           min_purity_increase=0,
#           n_subfeatures=0,
#           sampling_fraction=1.0,
#           )


# MODELS[:rfr] = (;
#                 :longname=>"Random Forest Regressor",
#                 :savename=>"RandomForestRegressor",
#                 :mdl=>rfr
#                 )



# 5. Fit each of the models to different subsets of features.
targets_to_try = [:CDOM, :CO, :Na, :Cl]


for target ∈ targets_to_try
    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]


    println("Working on $(target_name)")


    data_path = joinpath(datapath, target_name, "data")
    outpath = joinpath(datapath, target_name, "models")
    if !ispath(outpath)
        mkpath(outpath)
    end

    #X = CSV.read(joinpath(data_path, "X.csv"), DataFrame; types=Float32)
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



