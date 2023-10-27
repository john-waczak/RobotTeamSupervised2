using CSV, DataFrames
using MLJ
using Plots
using HDF5
using MLJFlux
using Flux
using HDF5
using ProgressMeter
include("mlp.jl")
include("utils.jl")



basepath = "/media/snuc/HSDATA/georectified/11/23/Scotty/"
modelpath ="/media/snuc/HSDATA/georectified/11/23/models/"

# basepath = "/media/john/HSDATA/georectified/11/23/Scotty/"
# modelpath ="/media/john/HSDATA/georectified/11/23/models/"



# Load in model ---------------------------------------------------

CR = @load ConstantRegressor pkg=MLJModels
DTR = @load DecisionTreeRegressor pkg=DecisionTree
DCR = @load DeterministicConstantRegressor pkg=MLJModels
ENR = @load ElasticNetRegressor pkg=MLJLinearModels
ETG = @load EvoTreeGaussian pkg=EvoTrees
ETR = @load EvoTreeRegressor pkg=EvoTrees
HR = @load HuberRegressor pkg=MLJLinearModels
KNNR = @load KNNRegressor pkg=NearestNeighborModels
LADR = @load LADRegressor pkg=MLJLinearModels
LGBMR = @load LGBMRegressor pkg=LightGBM
LR = @load LassoRegressor pkg=MLJLinearModels
LinearR = @load LinearRegressor pkg=MLJLinearModels
NNR = @load NeuralNetworkRegressor pkg=MLJFlux
QR = @load QuantileRegressor pkg=MLJLinearModels
RFR = @load RandomForestRegressor pkg=DecisionTree
RR = @load RidgeRegressor pkg=MLJLinearModels
RobustR = @load RobustRegressor pkg=MLJLinearModels
XGBR = @load XGBoostRegressor pkg=XGBoost
PCA = @load PCA pkg = MultivariateStats



mystack = Stack(;metalearner=LinearR(),
                resampling=CV(),
                cr = CR(),
                dtr = DTR(),
                dcr = DCR(),
                enr = ENR(),
                etg = ETG(),
                etr = ETR(nrounds=100),
                hr = HR(),
                #knnr = KNNR(),  # too memory intensive
                ladr = LADR(),
                lgbmr = LGBMR(), # too slow to train
                lr = LR(),
                nnr = NNR(builder=MLP((100, 50, 30), relu),
                          batch_size = 32,
                          optimiser=ADAM(0.01),
                          rng=42,
                          epochs=250,
                          ),
                qr = QR(),
                rfr = RFR(max_depth=30, n_trees=250, sampling_fraction=0.5),
                rr = RR(),
                robustr = RobustR(),
                #xgbr = XGBR(),
                )


pipe = @pipeline(Standardizer(),
                 PCA(pratio= 0.99999),
                 mystack,
                 target=Standardizer(),
                 name="Stacked Pipe",
                 )



targets = ["CO"]

for target ∈ targets
    # load in the model
    println("Loading model for $(target)")
    mach = machine(joinpath(modelpath, target, "stacked_"*target*".jlso"))

    for (root, dirs, files) ∈ walkdir(basepath)
        for f ∈ files
            if endswith(f, "df.csv")
                println("Working on $(root).")
                GC.gc()
                println("\tLoading df")
                df = DataFrame(CSV.File(joinpath(root, f)))
                # split off the inputs
                println("\tSplitting Inputs off")
                X = df[!, 3:971] # ilat and ilon are first
                Xfinal = X[!, Not([:MSR_705, :rad_MSR_705, :heading, :roll, :pitch])]

                println("\tPredicting...")
                pred = predict(mach, Xfinal);
                println("\tSuccess. Updating df")
                df[!, target*"_noOrientation"] = pred
                println("\tSaving")
                CSV.write(joinpath(root, f), df)
            end
        end
    end
end




