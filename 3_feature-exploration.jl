using CSV, DataFrames
using ProgressMeter
using Statistics
using Random

using Dates
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




datapath = "/Users/johnwaczak/data/robot-team/supervised"

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

# opt = Flux.Optimise.ADAM()
# # match sklearn.MLPRegressor default parameters
# nnr_mod = NNR(
#     builder=MLJFlux.MLP(hidden=(100,), σ=Flux.relu),
#     batch_size=200,
#     optimiser=Flux.Optimise.ADAM(0.001),
#     lambda = 0.0,
#     alpha = 0.0001,
#     rng=rng,
#     epochs=500,
# )

# nnr_mod = NNR(
# #     builder=MLJFlux.Short(n_hidden=250, dropout=0.1, σ=Flux.relu),
# #     batch_size=256,
# #     optimiser=Flux.Optimise.ADAM(),
# #     lambda=0,
# #     alpha=0,
# #     rng=rng,
# #     epochs=1000
# # )


# nnr  = Pipeline(
#     selector=FeatureSelector(features=Symbol.(cols_to_use)),
#     stand=Standardizer(features=Symbol.(cols_to_standardize)),
#     mdl=nnr_mod
# )


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
                :suffix => "vanilla",
                :mdl => ETR(nrounds=100, nbins=255, eta=0.3, max_depth=6, lambda=1.0, alpha=0.0),
                )


# MODELS[:dtr] = (;
#                 :longname => "Decision Tree Regressor",
#                 :savename => "DecisionTreeRegressor",
#                 :mdl => DTR()
#                 )


MODELS[:rfr] = (;
                 :longname => "Random Forest Regressor",
                 :savename => "RandomForestRegressor",
                 :packagename => "DecisionTree",
                 :suffix => "vanilla",
                 :mdl => RFR(n_subfeatures=-1, sampling_fraction=1.0, n_trees=150),
                 )



# MODELS[:rfr0] = (;
#                 :longname => "Random Forest Regressor",
#                 :savename => "RandomForestRegressor",
#                 :packagename => "DecisionTree",
#                 :suffix => "vanilla",
#                 :mdl => RFR(n_subfeatures=-1, sampling_fraction=0.9, n_trees=150),
#                 )

# MODELS[:rfr1] = (;
#                 :longname => "Random Forest Regressor",
#                 :savename => "RandomForestRegressor",
#                 :packagename => "DecisionTree",
#                 :suffix => "vanilla_200",
#                 :mdl => RFR(n_subfeatures=-1, sampling_fraction=0.9, n_trees=200),
#                 )


# MODELS[:rfr2] = (;
#                  :longname => "Random Forest Regressor",
#                  :savename => "RandomForestRegressor",
#                  :packagename => "DecisionTree",
#                  :suffix => "hpo1",
#                  :mdl => RFR(n_subfeatures=350, sampling_fraction=0.95, n_trees=85),
#                 )

# MODELS[:rfr3] = (;
#                  :longname => "Random Forest Regressor",
#                  :savename => "RandomForestRegressor",
#                  :packagename => "DecisionTree",
#                  :suffix => "hpo2",
#                  :mdl => RFR(n_subfeatures=350, sampling_fraction=0.95, n_trees=100),
#                  )


# 5. Fit each of the models to different subsets of features.
# targets_to_try = [:CDOM, :CO, :Na, :Cl]


# collections = ["11-23", "Full"]

targets_to_try = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490])]
#collections = ["11-23", "Full"]
collections = ["Full"]

# targets_to_try = [:CDOM,]

# collections = ["11-23",]

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

        # Xtrain = X[idx_train, :]
        # ytrain = y[idx_train]

        # Xtest = X[idx_test, :]
        # ytest = y[idx_test]

        for (shortname, model) ∈ MODELS
            try
                T1 = now()

                train_basic(
                    X, y,
                    idx_train, idx_test,
                    model.longname, model.savename, model.mdl,
                    target_name, units, target_long,
                    outpath;
                    suffix=model.suffix,
                    run_occam=false,
                )



                T2 = now()
                Δtrain = round((T2 - T1).value / 1000 / 60, digits=3) # min
                @warn "Training time: $(Δtrain) minutes"

                GC.gc()


                T1 = now()

                train_hpo(
                    X, y,
                    idx_train, idx_test,
                    model.longname, model.savename, model.packagename,
                    target_name, units, target_long,
                    model.mdl,
                    outpath;
                    nmodels=100
                )

                T2 = now()
                Δtrain = round((T2 - T1).value / 1000 / 60, digits=3) # min
                @warn "HPO Training time: $(Δtrain) minutes"

                GC.gc()
            catch e
                println("\t$(e)")
            end
        end


    end
end
GC.gc()






function make_summary_table(collection, savename, type, suffix)
    res_dicts = Dict[]

    for target in targets_to_try
        target_name = String(target)
        target_long = targets_dict[target][2]
        units = targets_dict[target][1]

        res_path = joinpath(datapath, collection, target_name, "models", savename, type, savename*"__$(suffix).json")

        @assert ispath(res_path)

        res_string = read(res_path, String)
        res_dict = JSON.parse(res_string)

        try
            for (key, val) in res_dict
                res_dict[key] = Float64(val)
            end
            res_dict["target"] = target_name
            res_dict["target_long"] = targets_dict[Symbol(target_name)][2]
            push!(res_dicts, res_dict)
        catch e
            println(target_name)
        end
    end

    df_res = DataFrame(res_dicts)

    sort!(df_res, :rsq_test; rev=true)

    select!(df_res, [:target_long, :target, :rsq_train, :rsq_test, :rmse_train, :rmse_test, :mae_train, :mae_test, :r_train, :r_test, :uncertainty, :emp_cov])
    rename!(df_res,
            "target_long" => "Target",
            "target" => "Var Name",
            "rsq_train" => "R² (train)",
            "rsq_test" => "R² (test)",
            "rmse_train" => "RMSE (train)",
            "rmse_test" => "RMSE (test)",
            "mae_train" => "MAE (train)",
            "mae_test" => "MAE (test)",
            "r_train" => "R (train)",
            "r_test" => "R (test)",
            "uncertainty" => "Estimated Uncertainty",
            "emp_cov" => "Empirical Coverage"
            )

    return df_res
end


function generate_tex_table(df)
    # target name (unit) # R^2 # RMSE # MAE # Uncertainty # Empirical Coverage #


    out = "\\begin{table}[H]\n"
    out = out * "  \\caption{This is a table caption.\\label{tab:fit-results}}\n"
    out = out * "  \\begin{adjustwidth}{-\\extralength}{0cm}\n"
    out = out * "  \\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n"
    out = out * "  \\begin{tabularx}{\\fulllength}{CCCCCCCCC}\n"
    out = out * "    \\toprule\n"
    out = out * "    \\textbf{Target (units)} & \\textbf{\$\\text{R}^2 (train)\$}	& \\textbf{\$\\text{R}^2 (test)\$}	& \\textbf{RMSE (train)} & \\textbf{RMSE (test)} & \\textbf{MAE (train)} & \\textbf{MAE (test)} & \\textbf{Estimated Uncertainty} & \\textbf{Empirical Coverage (\\%)}\\\\\n"
    out = out * "    \\midrule\n"

    for row in eachrow(df)
        target_tex = targets_latex[Symbol(row["Var Name"])]

        r2_train = string(round(row["R² (train)"], sigdigits=3))
        r2_test = string(round(row["R² (train)"], sigdigits=3))

        RMSE_train = string(round(row["RMSE (train)"], sigdigits=3))
        RMSE_test = string(round(row["RMSE (test)"], sigdigits=3))

        MAE_train = string(round(row["MAE (train)"], sigdigits=3))
        MAE_test = string(round(row["MAE (test)"], sigdigits=3))

        unc = " ± " * string(round(row["Estimated Uncertainty"], sigdigits=2))
        cov = string(round(row["Empirical Coverage"]*100, sigdigits=3))

        out = out * "    $(target_tex) & $(r2_train) & $(r2_test) & $(RMSE_train) & $(RMSE_test) & $(MAE_train) & $(MAE_test) & $(unc) & $(cov)\\\\\n"
        out = out * "    \\midrule\n"
    end

    out = out * "    \\bottomrule\n"
    out = out * "  \\end{tabularx}\n"
    out = out * "  \\end{adjustwidth}\n"
    out = out * "\\end{table}\n"

    return out
end







dfs = [
    make_summary_table(collections[1], "EvoTreeRegressor", "default", "vanilla"),
    make_summary_table(collections[1], "EvoTreeRegressor", "hyperparameter_optimized", "hpo"),
    make_summary_table(collections[1], "RandomForestRegressor", "default", "vanilla"),
    make_summary_table(collections[1], "RandomForestRegressor", "hyperparameter_optimized", "hpo"),
]



df_out = []

for target in targets_to_try
    idx_winner = argmax([df[df[:, "Var Name"] .== string(target), "R² (test)"]] for df in dfs)
    row = dfs[idx_winner][findfirst(dfs[idx_winner][:, "Var Name"] .== string(target)), :]

    df_row = DataFrame(row)
    df_row[!, "model"] = [models[idx_winner]]
    push!(df_out, df_row)
end

df_out = vcat(df_out...)
sort!(df_out, "R² (test)"; rev=true)

CSV.write(joinpath(datapath, "Full", "Summary-results.csv"), df_out)
tex_str = generate_tex_table(df_out)
open(joinpath(datapath, "Full", "Summary-results.tex"), "w") do f
    println(f, tex_str)
end




