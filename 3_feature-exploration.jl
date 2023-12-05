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

# add in ConformalPrediction via GitHub until my fix ships
using Pkg
Pkg.add(url="https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl.git")
using ConformalPrediction
using JSON

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
cols_to_standardize = ["roll", "pitch", "heading", "view_angle", "solar_azimuth", "solar_elevation", "solar_zenith"]
cols_to_use = vcat(cols_to_use, cols_to_standardize)




# #nnr_mod = NNR(builder=MLJFlux.MLP(hidden=(50,50,50,50,50), σ=Flux.relu),
# nnr_mod = NNR(builder=MLJFlux.MLP(hidden=(50,50,50,50), σ=Flux.relu),
#           batch_size=256,
#           optimiser=Flux.Optimise.ADAM(),
#           lambda = 0.0001,
#           rng=42,
#           epochs=500,
#           )


# nnr  = Pipeline(
#     selector=FeatureSelector(features=Symbol.(cols_to_use)),
#     stand=Standardizer(features=Symbol.(cols_to_standardize)),
#     mdl=nnr_mod
# )



# MODELS[:nnr] = (;
#                 :longname=>"Neural Network Regressor",
#                 :savename => "NeuralNetworkRegressor",
#                 #:mdl => Standardizer(features = name -> !(name ∈ ref_cols) ) |> nnr
#                 :mdl => nnr
#                 )



# 3. Add XGBoostRegressor. Defaults seem fine...
MODELS[:xgbr] = (;
                 :longname=>"XGBoost Regressor",
                 :savename=>"XGBoostRegressor",
                 :packagename=>"XGBoost",
                 :mdl=>XGBR(),
                 )


# MODELS[:etr] = (;
#                 :longname=>"Evo Tree Regressor",
#                 :savename=>"EvoTreeRegressor",
#                 :mdl => ETR()
#                 )

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

# targets_to_try = Symbol.(keys(targets_dict))

targets_to_try = [:CDOM]


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

    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]

    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]

    for (shortname, model) ∈ MODELS
        try
            train_basic(X, y,
                        Xtest, ytest,
                        model.longname, model.savename, model.mdl,
                        target_name, units, target_long,
                        outpath;
                        suffix="vanilla",
                        )

            GC.gc()

            # train_hpo(
            #     X, y,
            #     Xtest, ytest,
            #     model.longname, model.savename, model.packagename,
            #     target_name, units, target_long,
            #     model.mdl,
            #     outpath;
            #     nmodels=24
            # )

        catch e
            println("\t$(e)")
        end
    end

end

GC.gc()



# create summary table
res_dicts = Dict[]

for target in targets_to_try
    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    res_path = joinpath(datapath, target_name, "models", "XGBoostRegressor", "default", "XGBoostRegressor__vanilla.json")

    res_string = read(res_path, String)
    res_dict = JSON.parse(res_string)
    try
        for (key, val) in res_dict
            res_dict[key] = Float64(val)
        end

        res_dict["target"] = target_name
        push!(res_dicts, res_dict)
    catch e
        println(target_name)
    end
end

df_res = DataFrame(res_dicts)

sort!(df_res, :rsq_test; rev=true)
select!(df_res, [:target, :rsq_test, :rsq_train, :rmse_test, :rmse_train, :mae_test, :mae_train, :cov])
rename!(df_res,
        "target" => "Target",
        "rsq_test" => "R² test",
        "rsq_train" => "R² train",
        "rmse_test" => "RMSE test",
        "rmse_train" => "RMSE train",
        "mae_test" => "MAE test",
        "mae_train" => "MAE train",
        "cov" => "Empirical Coverage"
        )

CSV.write(joinpath(datapath, "vanilla_results.csv"), df_res)
