using CSV, DataFrames
using Plots, StatsPlots
using Random
using HDF5
using MLJ
using MLDataUtils
using MLJFlux
using ProgressMeter
using Statistics
include("mlp.jl")
include("utils.jl")
include("config.jl")
Random.seed!(42)


# load in training data
basepath = "/media/snuc/HSDATA/georectified/11/23/"
# basepath = "/media/john/HSDATA/georectified/11/23/"
ispath(basepath)

# for parallelism
MLJ.default_resource(CPUThreads())

PCA = @load PCA pkg = MultivariateStats
NN = @load NeuralNetworkRegressor pkg=MLJFlux
# 1 hidden layer with 50 hidden nodes
nn = NN(builder=MLJFlux.Short(n_hidden=100, σ=relu), epochs=25)

# build a Homogenous Ensemble of short NN
ensemble_model = EnsembleModel(atom=nn, n=20)


pipe = @pipeline(Standardizer(),
                 PCA(pratio= 0.9999),
                 ensemble_model,
                 target=Standardizer(),
                 name="Ensemble of NN",
                 )



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
    println("Outpath: ", outpath)


    targetsandfeatures = joinpath(basepath, "TargetsAndFeatures.csv")


    println("Grabbing training data...")
    Xtrain, ytrain, Xtest, ytest = getTrainingData(targetsandfeatures, target, 0.85, outpath)


    println("Fitting...")
    mach = machine(pipe, Xtrain, ytrain) |> fit!

    report(mach)


    println("Saving model...")
    MLJ.save(joinpath(outpath, "baggedNN_$(Symbol(target)).jlso"), mach)

    println("Plotting results...")
    p1_1, qqp = plotTrainingResults(mach,
                                    Xtrain, Xtest,
                                    ytrain, ytest,
                                    "$(targetName)",
                                    "Bag of Neural Networks")

    p1_1, qqp = plotTrainingResults(mach,
                                    Xtrain, Xtest,
                                    ytrain, ytest,
                                    "$(targetName)",
                                    "Bag of Neural Networks")



    savefig(p1_1, joinpath(outpath, "baggedNN_$(Symbol(target))_scatter.png"))
    savefig(qqp, joinpath(outpath, "baggedNN_$(Symbol(target))_qq.png"))

    println("Finished...")

end

