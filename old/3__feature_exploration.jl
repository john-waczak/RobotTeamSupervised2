using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter
using Interpolations
using Flux, MLJFlux

# using DecisionTree: impurity_importance

# set the plotting theme
add_mints_theme()
theme(:mints)

include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")

# MLJ.default_resource(CPUProcesses())
MLJ.default_resource(CPUThreads())

datapath = "/media/john/HSDATA/datasets/Full"
outpath = "/media/john/HSDATA/analysis_full"

isdir(datapath)
isdir(outpath)

RFR = @load RandomForestRegressor pkg=DecisionTree
XGBR = @load XGBoostRegressor pkg=XGBoost
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels

# need to add evotree regressor here and some other models.

# 1. Construct dictionary of models
MODELS = Dict()

# 2. Add Neural Network with defaults to mirror sk-learn
#    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
nnr = NNR(builder=MLJFlux.MLP(hidden=(250,100,50), σ=NNlib.relu),
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
    println("Working on $(String(target))"...)

    # load datasets
    target_name = String(target)
    target_long = targetsDict[target][2]
    units = targetsDict[target][1]

    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    # make sure to include the orientation stuff...
    others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]

    println("\tdataset loaded...")


    results_summary = []
    for (shortname, model) ∈ MODELS
        println("\t---- $(model.longname) ----")
        println("\ttraining reflectance only model...")
        # 1. Reflectance Only
        refs = [Symbol("λ_$(i)") for i ∈ 1:462]
        X̃ = X[:, vcat(refs, others)]
        X̃test = Xtest[:, vcat(refs, others)]

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="refs_only",
                              )
            push!(results_summary, res)
        catch e
            println("\t$(e)")
        end


        println("\ttraining reflectance + pseudo absorbance model...")
        # 2. Reflectance + Pseudo Absorbance
        for i ∈ 1:size(wavelengths, 1)
            pseudo = log.(1 ./ X̃[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃[!, "pseudo_absorb_$(i)"] = pseudo

            pseudo = log.(1 ./ X̃test[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃test[!, "pseudo_absorb_$(i)"] = pseudo
        end

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="refs_and_absorb",
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end


        println("\ttraining reflectance + derivative model...")
        # 3. Reflectance + Derivative
        X̃ = X[:, vcat(refs, others)]
        X̃test = Xtest[:, vcat(refs, others)]
        ΔR = zeros(size(X̃,1), size(wavelengths, 1))  # pre - allocate
        for i ∈ 1:nrow(X̃)
            rs = Array(X̃[i,refs])
            R = CubicSplineInterpolation(1:size(wavelengths, 1), rs)
            δR = only.(Interpolations.gradient.(Ref(R), 1:size(wavelengths, 1)))
            ΔR[i, :] .= δR
        end

        ΔRtest = zeros(size(X̃test, 1), size(wavelengths, 1))  # pre - allocate
        for i ∈ 1:nrow(X̃test)
            rs = Array(X̃test[i,refs])
            R = CubicSplineInterpolation(1:size(wavelengths, 1), rs)
            δR = only.(Interpolations.gradient.(Ref(R), 1:size(wavelengths, 1)))
            ΔRtest[i, :] .= δR
        end

        for i ∈ 1:size(wavelengths, 1)
            X̃[!, "ΔR_$(i)"] = ΔR[:,i]
            X̃test[!, "ΔR_$(i)"] = ΔRtest[:,i]
        end

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="refs_and_deriv",
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end



        println("\ttraining reflectance + derivative + pseudo absorbance model...")
        # 4. Reflectance + Derivative + Pseudo Absorbance
        for i ∈ 1:size(wavelengths, 1)
            pseudo = log.(1 ./ X̃[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃[!, "pseudo_absorb_$(i)"] = pseudo

            pseudo = log.(1 ./ X̃test[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃test[!, "pseudo_absorb_$(i)"] = pseudo
        end

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="refs_and_deriv_and_pseudo",
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end


        println("\ttraining radiance only model...")
        # 5. Radiance only
        rads = [Symbol("λ_$(i)_rad") for i ∈ 1:size(wavelengths, 1)]
        X̃ = X[:, vcat(rads, others)]
        X̃test = Xtest[:, vcat(rads, others)]

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="rads_only",
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end


        println("\ttraining radiance + downwelling model...")
        # 6. Radiance + Downwelling
        rads = [Symbol("λ_$(i)_rad") for i ∈ 1:size(wavelengths, 1)]
        downwells = [Symbol("λ_downwelling_$(i)") for i ∈ 1:size(downwelling_wavelengths, 1)]

        X̃ = X[:, vcat(rads, downwells, others)]
        X̃test = Xtest[:, vcat(rads, downwells, others)]

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix="rads_and_downwell",
                              )
            push!(results_summary, res)
        catch e
            println("\t$(e)")
        end


    end

    println("\tsaving results")
    res_df = DataFrame(results_summary)
    CSV.write(joinpath(outpath, "$(target_name)_model_comparison__fullfeatures.csv"), res_df)
end




# 6. Repeat, but using reduced features via importance ranking
N_features = 200

for target ∈ targets_to_try
    println("Working on $(String(target))"...)

    # load datasets
    target_name = String(target)
    target_long = targetsDict[target][2]
    units = targetsDict[target][1]

    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    # make sure to include the orientation stuff...
    others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]

    println("\tdataset loaded...")


    results_summary = []
    for (shortname, model) ∈ MODELS
        println("\t---- $(model.longname) ----")
        println("\ttraining reflectance only model...")



        # 1. Reflectance Only
        suf = "refs_only"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor", "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        refs = [Symbol("λ_$(i)") for i ∈ 1:462]
        X̃ = X[:, vcat(refs, others)]
        X̃test = Xtest[:, vcat(refs, others)]

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, pipe,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true
                              )
            push!(results_summary, res)
        catch e
            println("\t$(e)")
        end


        println("\ttraining reflectance + pseudo absorbance model...")
        # 2. Reflectance + Pseudo Absorbance

        suf = "refs_and_absorb"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor",  "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        for i ∈ 1:size(wavelengths, 1)
            pseudo = log.(1 ./ X̃[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃[!, "pseudo_absorb_$(i)"] = pseudo

            pseudo = log.(1 ./ X̃test[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃test[!, "pseudo_absorb_$(i)"] = pseudo
        end

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )


        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, pipe,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true,
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end


        println("\ttraining reflectance + derivative model...")
        # 3. Reflectance + Derivative
        suf = "refs_and_deriv"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor",  "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        X̃ = X[:, vcat(refs, others)]
        X̃test = Xtest[:, vcat(refs, others)]
        ΔR = zeros(size(X̃,1), size(wavelengths, 1))  # pre - allocate
        for i ∈ 1:nrow(X̃)
            rs = Array(X̃[i,refs])
            R = CubicSplineInterpolation(1:size(wavelengths, 1), rs)
            δR = only.(Interpolations.gradient.(Ref(R), 1:size(wavelengths, 1)))
            ΔR[i, :] .= δR
        end

        ΔRtest = zeros(size(X̃test, 1), size(wavelengths, 1))  # pre - allocate
        for i ∈ 1:nrow(X̃test)
            rs = Array(X̃test[i,refs])
            R = CubicSplineInterpolation(1:size(wavelengths, 1), rs)
            δR = only.(Interpolations.gradient.(Ref(R), 1:size(wavelengths, 1)))
            ΔRtest[i, :] .= δR
        end

        for i ∈ 1:size(wavelengths, 1)
            X̃[!, "ΔR_$(i)"] = ΔR[:,i]
            X̃test[!, "ΔR_$(i)"] = ΔRtest[:,i]
        end

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )


        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, pipe,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true,
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end



        println("\ttraining reflectance + derivative + pseudo absorbance model...")
        # 4. Reflectance + Derivative + Pseudo Absorbance
        suf = "refs_and_deriv_and_pseudo"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor",  "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        for i ∈ 1:size(wavelengths, 1)
            pseudo = log.(1 ./ X̃[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃[!, "pseudo_absorb_$(i)"] = pseudo

            pseudo = log.(1 ./ X̃test[!, "λ_$(i)"])
            pseudo .= min.(1e6, pseudo)
            X̃test[!, "pseudo_absorb_$(i)"] = pseudo
        end

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, pipe,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true,
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end

        println("\ttraining radiance only model...")
        # 5. Radiance only
        suf = "rads_only"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor",  "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        rads = [Symbol("λ_$(i)_rad") for i ∈ 1:size(wavelengths, 1)]
        X̃ = X[:, vcat(rads, others)]
        X̃test = Xtest[:, vcat(rads, others)]


        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, pipe,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true,
                              )
            push!(results_summary, res)

        catch e
            println("\t$(e)")
        end


        println("\ttraining radiance + downwelling model...")
        # 6. Radiance + Downwelling
        suf = "rads_and_downwell"
        fi_path = joinpath(outpath, String(target), "RandomForestRegressor",  "important_only","importance_ranking__$(suf).csv")
        fi_df = CSV.File(fi_path) |> DataFrame
        fi_df = fi_df[1:N_features, :]

        rads = [Symbol("λ_$(i)_rad") for i ∈ 1:size(wavelengths, 1)]
        downwells = [Symbol("λ_downwelling_$(i)") for i ∈ 1:size(downwelling_wavelengths, 1)]

        X̃ = X[:, vcat(rads, downwells, others)]
        X̃test = Xtest[:, vcat(rads, downwells, others)]

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
                        model=model.mdl
                        )

        try
            res = train_basic(y, X̃,
                              ytest, X̃test,
                              model.longname, model.savename, model.mdl,
                              target_name, units, target_long,
                              outpath;
                              suffix=suf,
                              features_reduced=true
                              )
            push!(results_summary, res)
        catch e
            println("\t$(e)")
        end


    end

    println("\tsaving results")
    res_df = DataFrame(results_summary)
    CSV.write(joinpath(outpath, "$(target_name)_model_comparison__reduced_features.csv"), res_df)
end




