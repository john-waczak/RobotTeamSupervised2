using CSV, DataFrames
using Plots, StatsPlots
using Random
using HDF5
using MLJ
using MLDataUtils
using ProgressMeter
using Statistics
using ShapML
include("config.jl")
include("mlp.jl")
include("utils.jl")
Random.seed!(42)

# for parallelism
MLJ.default_resource(CPUThreads())


# load in training data
basepath = "/media/snuc/HSDATA/georectified/11/23/"
# basepath = "/media/john/HSDATA/georectified/11/23/"
ispath(basepath)


DTR = @load DecisionTreeRegressor pkg=DecisionTree
PCA = @load PCA pkg = MultivariateStats

RFR = @load RandomForestRegressor pkg=DecisionTree
# dtr = DTR(post_prune=true, rng=42)
# bagOfTrees = EnsembleModel(atom=dtr, n=100,  bagging_fraction=0.7)

pca = PCA(pratio= 0.999)

# don't standardize inputs but do standardize outputs
# pipe = @pipeline(#Standardizer(),
#                  #pca,
#                  RFR(n_trees=200, rng=42),
#                  target=Standardizer(),
#                  name="RFR Pipe",
#                  )

rfr = RFR()
pipe = @pipeline(#Standardizer(),
                 #pca,
                 rfr,
                 target=Standardizer(),
                 name="Tuned RFR Pipe",
                 )

r0 = range(pipe, :(random_forest_regressor.n_trees), lower=10, upper=250)
#r1 = range(pipe, :(random_forest_regressor.max_depth), lower=2, upper=100);
r2 = range(pipe, :(random_forest_regressor.min_samples_leaf), lower=1, upper=20);
r3 = range(pipe, :(random_forest_regressor.min_samples_split), lower=2, upper=100);


tuned_rfr = TunedModel(model=pipe,
                       tuning=RandomSearch(rng=42),
                       resampling=CV(),
                       # range=[r0, r1, r2, r3],
                       range=[r0, r2, r3],
                       n = 500,
                       measure=rms);






for (key, val) ∈ targetsDict
    GC.gc()
    println("Working on $(key)...")

    target = key
    targetUnits = val[1]
    targetName = val[2]

    outpath = joinpath(basepath, "models", "$(Symbol(target))")
    if !ispath(outpath)
        mkdir(outpath)
    end
    println("\tOutpath: ", outpath)


    targetsandfeatures = joinpath(basepath, "TargetsAndFeatures.csv")


    println("\tGrabbing training data...")
    Xtrain, ytrain, Xtest, ytest = getTrainingData(targetsandfeatures, target, 0.85, outpath)


    # try training on only reflectance values
    λs = [Symbol("λ_$(i)") for i ∈ 1:462]
    λs_rad = [Symbol("λ_$(i)_rad") for i ∈ 1:462]
    Xtrain = Xtrain[!, Not(λs)]
    Xtest = Xtest[!, Not(λs)]




    println("\tFitting...")
    mach = machine(tuned_rfr, Xtrain, ytrain) |> fit!


    println("\tSaving best parameters")
    open(joinpath(outpath, "rfrhpo_fit_res.txt"), "a") do f
        show(f, MIME"text/plain"(), fitted_params(mach).best_model)
    end




    report(mach)


    println("\tSaving model...")
    MLJ.save(joinpath(outpath, "rfr_$(Symbol(target)).jlso"), mach)

    println("\tPlotting results...")
    p1_1, qqp = plotTrainingResults(mach,
                                    Xtrain, Xtest,
                                    ytrain, ytest,
                                    "$(targetName)",
                                    "Random Forest Regressor")

    savefig(p1_1, joinpath(outpath, "rfr_$(Symbol(target))_scatter.png"))
    savefig(p1_1, joinpath(outpath, "rfr_$(Symbol(target))_scatter.svg"))
    savefig(qqp, joinpath(outpath, "rfr_$(Symbol(target))_qq.png"))
    savefig(qqp, joinpath(outpath, "rfr_$(Symbol(target))_qq.svg"))

    println("\tRanking feature importance")
    # add the SHAP value calculation here
    function predict_function(model, data)
        data_pred = DataFrame(y_pred = MLJ.predict(model, data))
        return data_pred
    end

    Nexplain = 300
    sample_size = 60
    explain = Xtrain[1:Nexplain, :]
    # Compute stochastic Shapley values.
    data_shap = ShapML.shap(explain = explain,
                            model = mach,
                            predict_function = predict_function,
                            sample_size = sample_size,
                            seed = 42
                            )
    # group by feature name
    g_datashap = groupby(data_shap, :feature_name)
    # aggregate by mean abs shap value
    data_plot = combine(g_datashap, :shap_effect => (x->mean(abs.(x))) => :mean_effect)
    # sort by mean_effect
    idx = sortperm(data_plot.mean_effect)

    plot_features = data_plot.feature_name[idx]
    plot_vals = data_plot.mean_effect[idx]
    pbar = bar(plot_features[1:20], plot_vals[1:20] , yticks=(0.5:1:19.5, plot_features[1:20]), orientation = :horizontal, label="")
    xlabel!("Mean Absolute SHAP value")
    title!("Ranked Feature Importance for $(targetName)")
    savefig(pbar, joinpath(outpath, "rfr_$(Symbol(target))_featureImportance.png"))
    savefig(pbar, joinpath(outpath, "rfr_$(Symbol(target))_featureImportance.svg"))


    println("\tFinished...")

end


