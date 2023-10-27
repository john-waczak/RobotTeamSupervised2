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


DTR = @load DecisionTreeRegressor pkg=DecisionTree
PCA = @load PCA pkg = MultivariateStats
RFR = @load RandomForestRegressor pkg=DecisionTree



# load in training data
# basepath = "/media/snuc/HSDATA/georectified/11/23/"
basepath = "/media/john/HSDATA/georectified/11/23/"
ispath(basepath)

# first let's hyperparameter optimize a single decision tree.
targetsDict
target = :CDOM
targetUnits, targetName, targetMin = targetsDict[target]



# collect the data
outpath = joinpath(basepath, "models", "$(Symbol(target))")
if !ispath(outpath)
    mkdir(outpath)
end

targetsandfeatures = joinpath(basepath, "TargetsAndFeatures.csv")
Xtrain, ytrain, Xtest, ytest = getTrainingData(targetsandfeatures, target, 0.85, outpath)

# try training on only reflectance values
λs = [Symbol("λ_$(i)") for i ∈ 1:462]
λs_rad = [Symbol("λ_$(i)_rad") for i ∈ 1:462]
Xtrain = Xtrain[!, Not(λs)]
Xtest = Xtest[!, Not(λs)]


rfr = RFR()
pipe = @pipeline(#Standardizer(),
                 #pca,
                 rfr,
                 target=Standardizer(),
                 name="Tuned RFR Pipe",
                 )

r0 = range(pipe, :(random_forest_regressor.n_trees), lower=10, upper=250)
r1 = range(pipe, :(random_forest_regressor.max_depth), lower=2, upper=100);
r2 = range(pipe, :(random_forest_regressor.min_samples_leaf), lower=1, upper=20);
r3 = range(pipe, :(random_forest_regressor.min_samples_split), lower=2, upper=100);


tuned_dtr = TunedModel(model=pipe,
                       tuning=RandomSearch(rng=42),
                       resampling=CV(nfolds=3),
                       range=[r0, r1, r2, r3],
                       n = 500,
                       measure=rms);



mach = machine(tuned_dtr, Xtrain, ytrain);
fit!(mach, verbosity=1);


fitted_params(mach).best_model

open("fit_res.txt", "a") do f
    show(f, MIME"text/plain"(), fitted_params(mach).best_model)
end




