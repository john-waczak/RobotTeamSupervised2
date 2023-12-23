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
using MLJ, Flux
using MLJBase: train_test_pairs

# add in ConformalPrediction via GitHub until my fix ships
# using Pkg
# Pkg.add(url="https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl.git")
using ConformalPrediction
using JSON

# using ShapML


# for base model prediction:
# using MLJModelInterface: reformat
# predict(conf_model.model, mach.fitresult, reformat(conf_model.model, Xtest)...)




# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


# pull in targets info
include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")




datapath = "/media/teamlary/LabData/RobotTeam/supervised"

# DecisionTreeRegressor and RandomForestRegressor don't seem to work.


RFR = @load RandomForestRegressor pkg=DecisionTree
DTR = @load DecisionTreeRegressor pkg=DecisionTree
XGBR = @load XGBoostRegressor pkg=XGBoost
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels
ETR = @load EvoTreeRegressor pkg=EvoTrees


# 1. Construct dictionary of models
MODELS = Dict()


cols_to_use = ["R_" * lpad(i, 3, "0") for i in 1:462]
cols_to_standardize = ["roll", "pitch", "heading", "view_angle", "solar_azimuth", "solar_elevation", "solar_zenith", "Σrad", "Σdownwelling"]
cols_to_use = vcat(cols_to_use, cols_to_standardize)

use_metrics = false


#nnr_mod = NNR(builder=MLJFlux.MLP(hidden=(50,50,50,50,50), σ=Flux.relu)
# nnr_mod = NNR(builder=MLJFlux.MLP(hidden=(50,50,50,50), σ=Flux.relu)

opt = Flux.Optimise.ADAM()
# match sklearn.MLPRegressor default parameters
nnr_mod = NNR(
    builder=MLJFlux.MLP(hidden=(100,), σ=Flux.relu),
    batch_size=200,
    optimiser=Flux.Optimise.ADAM(0.001),
    lambda = 0.0,
    alpha = 0.0001,
    rng=rng,
    epochs=500,
)

# nnr_mod = NNR(
# #     builder=MLJFlux.Short(n_hidden=250, dropout=0.1, σ=Flux.relu),
# #     batch_size=256,
# #     optimiser=Flux.Optimise.ADAM(),
# #     lambda=0,
# #     alpha=0,
# #     rng=rng,
# #     epochs=1000
# # )


nnr  = Pipeline(
    selector=FeatureSelector(features=Symbol.(cols_to_use)),
    stand=Standardizer(features=Symbol.(cols_to_standardize)),
    mdl=nnr_mod
)


# MODELS[:nnr] = (;
#                 :longname=>"Neural Network Regressor",
#                 :savename => "NeuralNetworkRegressor",
#                 :mdl => nnr
#                 )


# # # 3. Add XGBoostRegressor. Defaults seem fine...

# MODELS[:xgbr] = (;
#                  :longname=>"XGBoost Regressor",
#                  :savename=>"XGBoostRegressor",
#                  :packagename=>"XGBoost",
#                  :mdl=>XGBR(),
#                  )

MODELS[:etr] = (;
                :longname=>"Evo Tree Regressor",
                :savename=>"EvoTreeRegressor",
                :packagename => "EvoTrees",
                :mdl => ETR(nrounds=100, nbins=255, eta=0.3, max_depth=6, lambda=1.0, alpha=0.0),
                )


# MODELS[:dtr] = (;
#                 :longname => "Decision Tree Regressor",
#                 :savename => "DecisionTreeRegressor",
#                 :mdl => DTR()
#                 )


# MODELS[:rfr] = (;
#                 :longname => "Random Forest Regressor",
#                 :savename => "RandomForestRegressor",
#                 :mdl => RFR()
#                 )



# 5. Fit each of the models to different subsets of features.
# targets_to_try = [:CDOM, :CO, :Na, :Cl]


# collections = ["11-23", "Full"]

collections = ["11-23", "Full"]
targets_to_try = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490])]


for collection ∈ collections
    for target ∈ targets_to_try
        target_name = String(target)
        target_long = targets_dict[target][2]
        units = targets_dict[target][1]


        println("Working on $(target_name)")

        data_path = joinpath(datapath, collection, target_name, "data")
        outpath = joinpath(datapath, collection, target_name, "models")
        if !ispath(outpath)
            mkpath(outpath)
        end

        X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
        y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
        idx_train = CSV.read(joinpath(data_path, "idx_train.csv"), DataFrame)[:,1]
        idx_test= CSV.read(joinpath(data_path, "idx_test.csv"), DataFrame)[:,1]

        Xtrain = X[idx_train, :]
        ytrain = y[idx_train]

        Xtest = X[idx_test, :]
        ytest = y[idx_test]

        for (shortname, model) ∈ MODELS
            try
                train_folds(
                    Xtrain, ytrain,
                    Xtest, ytest,
                    model.longname, model.savename, model.mdl,
                    target_name, units, target_long,
                    outpath;
                    suffix="vanilla",
                )

                GC.gc()
            catch e
                println("\t$(e)")
            end
        end


    end
end
GC.gc()




function make_summary_table(modelname, ending)
    res_dicts = Dict[]

    for target in targets_to_try
        target_name = String(target)
        target_long = targets_dict[target][2]
        units = targets_dict[target][1]

        res_path = joinpath(datapath, target_name, "models", modelname, "default", modelname*"__vanilla"*ending*".json")

        res_string = read(res_path, String)

        res_dict = JSON.parse(res_string)

        try
            for (key, val) in res_dict
                res_dict[key] = Float64(val)
            end
            res_dict["target"] = target_name
            res_dict["target_long"] = targets_dict[Symbol(target_name)][2]
            push!(res_dicts, res_dict
        catch e
            println(target_name)
        end
    end

    df_res = DataFrame(res_dicts)

    sort!(df_res, :rsq_test; rev=true)
    select!(df_res, [:target_long, :target, :rsq_test, :rsq_train, :rmse_test, :rmse_train, :mae_test, :mae_train, :cov])
    rename!(df_res,
            "target_long" => "Target",
            "target" => "Var Name",
            "rsq_test" => "R² test",
            "rsq_train" => "R² train",
            "rmse_test" => "RMSE test",
            "rmse_train" => "RMSE train",
            "mae_test" => "MAE test",
            "mae_train" => "MAE train",
            "r_test" => "R test",
            "r_train" => "R train",
            "cov" => "Empirical Coverage"
            )

    return df_res
end


df_xgbr = make_summary_table("XGBoostRegressor", "")
df_xgbr_no_metrics = make_summary_table("XGBoostRegressor", "-no-metrics")
df_evtr = make_summary_table("EvoTreeRegressor", "")
df_evtr_no_metrics = make_summary_table("EvoTreeRegressor", "-no-metrics")
df_rfr = make_summary_table("RandomForestRegressor", "")
df_rfr_no_metrics = make_summary_table("RandomForestRegressor", "-no-metrics")



CSV.write(joinpath(datapath, "XGBoostRegressor-vanilla_results.csv"), df_xgbr)
CSV.write(joinpath(datapath, "XGBoostRegressor-vanilla-no-metrics_results.csv"), df_xgbr_no_metrics)
CSV.write(joinpath(datapath, "EvoTreeRegressor-vanilla_results.csv"), df_evtr)
CSV.write(joinpath(datapath, "EvoTreeRegressor-vanilla-no-metrics_results.csv"), df_evtr_no_metrics)
CSV.write(joinpath(datapath, "RandomForestRegressor-vanilla_results.csv"), df_rfr)
CSV.write(joinpath(datapath, "RandomForestRegressor-vanilla-no-metrics_results.csv"), df_rfr_no_metrics)



df_compare = DataFrame()
df_compare[:, "Var Name"] = df_xgbr[:, "Var Name"]
df_compare[:, "R² xgbr"] = df_xgbr[:, "R² test"]
df_compare[:, "R² xgbr no metrics"] = zeros(nrow(df_compare))
df_compare[:, "R² evtr"] = zeros(nrow(df_compare))
df_compare[:, "R² evtr no metrics"] = zeros(nrow(df_compare))
df_compare[:, "R² rfr"] = zeros(nrow(df_compare))
df_compare[:, "R² rfr no metrics"] = zeros(nrow(df_compare))


for row in eachrow(df_compare)
    row["R² xgbr no metrics"] = df_xgbr_no_metrics[df_xgbr_no_metrics[!,"Var Name"] .== row["Var Name"], "R² test"][1]
    row["R² evtr"] = df_evtr[df_evtr[!,"Var Name"] .== row["Var Name"], "R² test"][1]
    row["R² evtr no metrics"] = df_evtr_no_metrics[df_evtr_no_metrics[!,"Var Name"] .== row["Var Name"], "R² test"][1]
    row["R² rfr"] = df_rfr[df_rfr[!,"Var Name"] .== row["Var Name"], "R² test"][1]
    row["R² rfr no metrics"] = df_rfr_no_metrics[df_rfr_no_metrics[!,"Var Name"] .== row["Var Name"], "R² test"][1]
end

df_compare



