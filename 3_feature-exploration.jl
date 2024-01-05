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


ETR = @load EvoTreeRegressor pkg=EvoTrees
RFR = @load RandomForestRegressor pkg=DecisionTree
DTR = @load DecisionTreeRegressor pkg=DecisionTree
XGBR = @load XGBoostRegressor pkg=XGBoost
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels


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
#                  :suffix => "vanilla",
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
#                 :suffix => "vanilla",
#                 :mdl => DTR()
#                 )


MODELS[:rfr] = (;
                 :longname => "Random Forest Regressor",
                 :savename => "RandomForestRegressor",
                 :packagename => "DecisionTree",
                 :suffix => "vanilla",
                 :mdl => RFR(n_subfeatures=-1, sampling_fraction=1.0, n_trees=150),
                 )



# 5. Fit each of the models to different subsets of features.

collections = ["11-23", "Full"]
# collections = ["Full"]


targets_to_try = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490])]

# targets_to_try = [:CDOM,:CO]
# targets_to_try = [:CDOM,]

# targets_to_try = [:Chl]

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


        for (shortname, model) ∈ MODELS
            try
                T1 = now()

                train_folds(
                    X, y,
                    idx_train, idx_test,
                    model.longname, model.savename, model.mdl,
                    target_name, units, target_long,
                    outpath;
                    suffix=model.suffix,
                    nfolds=10,
                )


                T2 = now()
                Δtrain = round((T2 - T1).value / 1000 / 60, digits=3) # min
                @warn "Training time: $(Δtrain) minutes"

                GC.gc()

                # for each of these models, let's now load in
                # the most important features and do hyperparameter
                # optimization

                fi_path = joinpath(outpath, model.savename, "default", "importance_ranking__$(model.suffix).csv")
                fi_df = CSV.read(fi_path, DataFrame)
                N_final = 25
                fi_n = @view fi_df[1:N_final, :]
                fi_occam = Symbol.(fi_n.feature_name)

                X_occam = X[:, fi_occam]

                T1 = now()

                train_hpo(
                    X_occam, y,
                    idx_train, idx_test,
                    model.longname, model.savename, model.packagename,
                    target_name, units, target_long,
                    model.mdl,
                    outpath;
                    nfolds=10,
                    nmodels=100,
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







# now we need to revert the summary table to the old version... lol
function make_summary_table(collection, savename, type, suffix)
    res_dicts = Dict[]

    for target in targets_to_try
        target_name = String(target)
        target_long = targets_dict[target][2]
        units = targets_dict[target][1]

        res_path = joinpath(datapath, collection, target_name, "models", savename, type, savename * suffix * ".json")
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

    sort!(df_res, :rsq_mean; rev=true)

    select!(df_res, [:target_long, :target, :rsq_mean, :rsq_std, :rmse_mean, :rmse_std, :mae_mean, :mae_std, :r_mean, :r_std, :uncertainty, :emp_cov])
    rename!(df_res,
            "target_long" => "Target",
            "target" => "Var Name",
            "rsq_mean" => "R² (mean)",
            "rsq_std" => "R² (std)",
            "rmse_mean" => "RMSE (mean)",
            "rmse_std" => "RMSE (std)",
            "mae_mean" => "MAE (mean)",
            "mae_std" => "MAE (std)",
            "r_mean" => "R (mean)",
            "r_std" => "R (std)",
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
    out = out * "  \\begin{tabularx}{\\fulllength}{CCCCCC}\n"
    out = out * "    \\toprule\n"
    out = out * "    \\textbf{Target (units)} & \\textbf{\$\\text{R}^2\$} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{Estimated Uncertainty} & \\textbf{Empirical Coverage (\\%)}\\\\\n"
    out = out * "    \\midrule\n"

    for row in eachrow(df)
        target_tex = targets_latex[Symbol(row["Var Name"])]

        r2_str = string(round(row["R² (mean)"], sigdigits=3)) * " ± " * string(round(row["R² (std)"], sigdigits=3))
        rmse_str = string(round(row["RMSE (mean)"], sigdigits=3)) * " ± " * string(round(row["RMSE (std)"], sigdigits=3))
        mae_str = string(round(row["MAE (mean)"], sigdigits=3)) * " ± " * string(round(row["MAE (std)"], sigdigits=3))
        # r_str = string(round(row["R (mean)"], sigdigits=3)) * " ± " * string(round(row["R (std)"], sigdigits=3))
        unc = " ± " * string(round(row["Estimated Uncertainty"], sigdigits=2))
        cov = string(round(Int, row["Empirical Coverage"]*100))

        out = out * "    $(target_tex) & $(r2_str) & $(rmse_str) & $(mae_str) & $(unc) & $(cov)\\\\\n"
        out = out * "    \\midrule\n"
    end

    out = out * "    \\bottomrule\n"
    out = out * "  \\end{tabularx}\n"
    out = out * "  \\end{adjustwidth}\n"
    out = out * "\\end{table}\n"

    return out
end




collections[1]
res_path = joinpath(datapath, collections[1], "CDOM", "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"-occam__vanilla.json")
ispath(res_path)

dfs = [
    #make_summary_table(collections[1], "EvoTreeRegressor", "default", "-occam__vanilla"),
    make_summary_table(collections[1], "EvoTreeRegressor", "hyperparameter_optimized", "__hpo"),
    #make_summary_table(collections[1], "RandomForestRegressor", "default", "-occam__vanilla"),
    make_summary_table(collections[1], "RandomForestRegressor", "hyperparameter_optimized", "__hpo"),
]



model_ids = ["etr hpo", "rfr hpo"]

df_out = []


for target in targets_to_try
    idx_winner = argmax([df[df[:, "Var Name"] .== string(target), "R² (mean)"] for df in dfs])
    row = dfs[idx_winner][findfirst(dfs[idx_winner][:, "Var Name"] .== string(target)), :]

    df_row = DataFrame(row)
    df_row[!, "model"] = [model_ids[idx_winner]]
    push!(df_out, df_row)
end


df_out = vcat(df_out...)
sort!(df_out, "R² (mean)"; rev=true)

df_out[:, ["Var Name", "R² (mean)", "RMSE (mean)", "Estimated Uncertainty", "Empirical Coverage"]]

CSV.write(joinpath(datapath, "Full", "Summary-results.csv"), df_out)
tex_str = generate_tex_table(df_out)
open(joinpath(datapath, "Full", "Summary-results.tex"), "w") do f
    println(f, tex_str)
end




