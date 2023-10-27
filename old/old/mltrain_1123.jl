using CSV, DataFrames
using Plots, StatsPlots
using Random
using HDF5
using MLJ
using MLDataUtils
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


# load in models
bf = 0.7
PCA = @load PCA pkg = MultivariateStats

CR = @load ConstantRegressor pkg=MLJModels

DTR = @load DecisionTreeRegressor pkg=DecisionTree
# ensemble_dtr = EnsembleModel(atom=DTR(post_prune=true, rng=42), n=100, bagging_fraction=bf)

RFR = @load RandomForestRegressor pkg=DecisionTree

DCR = @load DeterministicConstantRegressor pkg=MLJModels

ENR = @load ElasticNetRegressor pkg=MLJLinearModels

ETG = @load EvoTreeGaussian pkg=EvoTrees

ETR = @load EvoTreeRegressor pkg=EvoTrees

HR = @load HuberRegressor pkg=MLJLinearModels

KNNR = @load KNNRegressor pkg=NearestNeighborModels
knnr = KNNR(K=10)

LADR = @load LADRegressor pkg=MLJLinearModels

LGBMR = @load LGBMRegressor pkg=LightGBM

LR = @load LassoRegressor pkg=MLJLinearModels

LinearR = @load LinearRegressor pkg=MLJLinearModels

NNR = @load NeuralNetworkRegressor pkg=MLJFlux
# nn = NNR(builder=MLJFlux.Short(n_hidden=50, σ=relu), epochs=25)
# ensemble_nn = EnsembleModel(atom=nn, n=20, bagging_fraction=bf)
nnr = NNR(builder=MLP((30, 30, 30, 30, 30), relu),
          batch_size = 32,
          optimiser=ADAM(0.01),
          rng=42,
          epochs=250,
          )


QR = @load QuantileRegressor pkg=MLJLinearModels

RR = @load RidgeRegressor pkg=MLJLinearModels

RobustR = @load RobustRegressor pkg=MLJLinearModels



mystack = Stack(;metalearner=LinearR(),
                resampling=CV(nfolds=3, shuffle=true, rng=42),
                cr = CR(),
                rfr = RFR(n_trees=200, rng=42),
                dcr = DCR(),
                enr = ENR(),
                etg = ETG(),
                hr = HR(),
                knnr = knnr,
                ladr = LADR(),
                lr = LR(),
                linear = LinearR(),
                nn = nnr,
                qr = QR(),
                robustr = RobustR(),
                )


pipe = @pipeline(#Standardizer(),
                 #PCA(pratio= 0.999),
                 mystack,
                 target=Standardizer(),
                 name="Stacked Pipe",
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


    # try training on only reflectance values
    λs = [Symbol("λ_$(i)") for i ∈ 1:462]
    λs_rad = [Symbol("λ_$(i)_rad") for i ∈ 1:462]
    Xtrain = Xtrain[!, Not(λs)]
    Xtest = Xtest[!, Not(λs)]


    try
        println("Fitting...")
        mach = machine(pipe, Xtrain, ytrain) |> fit!

        report(mach)


        println("Saving model...")
        MLJ.save(joinpath(outpath, "superlearner_$(Symbol(target)).jlso"), mach)

        println("Plotting results...")
        p1_1, qqp = plotTrainingResults(mach,
                                        Xtrain, Xtest,
                                        ytrain, ytest,
                                        "$(targetName)",
                                        "Super Learner")

        p1_1, qqp = plotTrainingResults(mach,
                                        Xtrain, Xtest,
                                        ytrain, ytest,
                                        "$(targetName)",
                                        "Super Learner")



        savefig(p1_1, joinpath(outpath, "superLearner_$(Symbol(target))_scatter.png"))
        savefig(p1_1, joinpath(outpath, "superLearner_$(Symbol(target))_scatter.svg"))
        savefig(qqp, joinpath(outpath, "superLearner_$(Symbol(target))_qq.png"))
        savefig(qqp, joinpath(outpath, "superLearner_$(Symbol(target))_qq.svg"))

        println("Finished...")

    catch e
        println("Unable to fit model")
        println(e)
    end
end




