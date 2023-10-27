using StableRNGs
using JSON

# seed rng for reproducibility
rng = StableRNG(42)



function train_basic(y, X̃,
                     ytest, X̃test,
                     longname, savename, mdl,
                     target_name, units, target_long,
                     outpath;
                     suffix="",
                     predict_function = MLJ.predict,
                      )

    #######################################################################################
    #                        0. Set up outgoing paths                                     #
    #######################################################################################
    outpathtarget = joinpath(outpath, target_name)
    outpathmodel = joinpath(outpathtarget, savename)
    outpathdefault = joinpath(outpathmodel, "default")
    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
#    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")

    path_to_use = outpathdefault

    for path ∈ [outpathtarget, outpathmodel, outpathdefault, outpath_featuresreduced] # , outpath_hpo]
        if !isdir(path)
            mkpath(path) end
    end

    scitype(y) <: target_scitype(mdl)
    scitype(X̃) <: input_scitype(mdl)

    #######################################################################################
    #   1. Train model
    #######################################################################################
    try
        mdl.rng = rng
    catch
        println("\t$(longname) does not have parameter :rng")
    end

    mach = machine(mdl, X̃, y)
    fit!(mach)


    ŷ = predict_function(mach, X̃)  # generate predictions on training set
    ŷtest = predict_function(mach, X̃test)  # generate predictions on testing set


    p1 = scatterresult(y, ŷ,
                      ytest, ŷtest;
                      xlabel="True $(target_long) [$(units)]",
                      ylabel="Predicted $(target_long) [$(units)]",
                      plot_title="Fit for $(longname)",)

    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).png"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).svg"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).pdf"))

    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          xlabel="True $(target_long) [$(units)]",
                          ylabel="Predicted $(target_long) [$(units)]",
                          title="Fit for $(longname)",)

    savefig(p2, joinpath(path_to_use, "qq__$(suffix).png"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).svg"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).pdf"))


    # save the model
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)

    #######################################################################################
    #   2. Compute feature importances
    #######################################################################################
    if reports_feature_importances(mdl)
        println("\tcomputing feature importances...")

        fi_pairs = feature_importances(mach)  # `:impurity` feature importances
        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
        fi_df.plot_name = String.(fi_df.feature_name)


        if name_replacements != nothing
            for row ∈ eachrow(fi_df)
                fname = String(row.feature_name)
                if fname ∈ keys(name_replacements)
                    row.plot_name = name_replacements[fname]
                end
            end
        end

        pfi = rankimportances(fi_df.plot_name[1:25], fi_df.rel_importance[1:25]; ylabel="Normalized Impurity Importance", title="Ranked Feature Importance", xtickfontsize=7)

        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).png"))
        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).svg"))
        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).pdf"))


        CSV.write(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).csv"), fi_df)
    end

    return rsq(ŷ, y), rsq(ŷtest, ytest), rmse(ŷtest, ytest), mae(ŷtest, ytest)
end







# define dictionary with default hyperparameter values used for tuning for each model
# use a dict of dicts structure

# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook

hpo_ranges = Dict("DecisionTree" => Dict("DecisionTreeRegressor" => [(hpname=:min_samples_leaf, lower=2, upper=100),
                                                                   #   (hpname=:n_subfeatures, values=[-1,0]),
                                                                      (hpname=:max_depth, values=[-1, 2, 3, 5, 10, 20]),
                                                                      (hpname=:post_prune, values=[false, true])
                                                                      ],
                                         "RandomForestRegressor" => [
                                                                      (hpname=:min_samples_leaf, lower=2, upper=100),
                                                                      (hpname=:max_depth, values=[-1, 2, 3, 5, 10, 20]),
                                                                      (hpname=:n_subfeatures, values=[-1,0]),
                                                                   #  (hpname=:n_trees, lower=10, upper=100),
                                                                      (hpname=:n_trees, values=[10, 25, 50, 75, 100, 125, 150]),
                                                                      (hpname=:sampling_fraction, lower=0.65, upper=0.9)
                                                                      ],
                                          ),
                  "XGBoost" => Dict("XGBoostRegressor" => [(hpname=:eta, lower=0.01, upper=0.2),
                                                           (hpname=:gamma, lower=0, upper=100),  # not sure about this one
                                                           (hpname=:max_depth, lower=3, upper=10),
                                                           (hpname=:min_child_weight, lower=0.0, upper=5.0),
                                                           (hpname=:max_delta_step, lower=1.0, upper=10.0),
                                                           (hpname=:subsample, lower=0.5, upper=1.0),
                                                           (hpname=:lambda, lower=0.1, upper=5.0),  # L2 regularization. Higher makes model more conservative
                                                           (hpname=:alpha, lower=0.0, upper=1.0), # L1 regularization. Higher makes model more sparse
                                                           ],
                                    ),
                  "EvoTrees" => Dict("EvoTreeRegressor" => [(hpname=:nrounds, lower=10, upper=100),
                                                            (hpname=:eta, lower=0.01, upper=0.2),
                                                            (hpname=:gamma, lower=0, upper=100),  # not sure about this one
                                                            (hpname=:max_depth, lower=3, upper=10),
                                                            (hpname=:min_weight, lower=0.0, upper=5.0),
                                                            (hpname=:lambda, lower=0.1, upper=5.0),  # L2 regularization. Higher makes model more conservative
                                                            (hpname=:alpha, lower=0.0, upper=1.0), # L1 regularization. Higher makes model more sparse
                                                            ],
                                     ),
                  "NearestNeighborModels" => Dict("KNNRegressor" => [(hpname=:K, lower=1, upper=50),
                                                                     (hpname=:leafsize, lower=1, upper=50),
                                                                     ],
                                                  ),
                  "MLJFlux" => Dict("NeuralNetworkRegressor" =>[],
                                    ),
                  "LightGBM" => Dict("LGBMRegressor" => [(hpname=:num_iterations, lower=5, upper=100),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.3),
                                                         (hpname=:max_depth, lower=3, upper=12),
                                                         (hpname=:bagging_fraction, lower=0.65, upper=1.0),
                                                         (hpname=:bagging_freq, values=[1]),
                                                         ])
                  )



function train_hpo(y, X,
                   ytest, Xtest,
                   longname, savename, packagename, mdl,
                   target_name, units, target_long,
                   outpath;
                   nmodels = 20,
                   accelerate = true,
                   )

    suffix = "hpo"

    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpathmodel = joinpath(outpathtarget, savename)
    if !isdir(outpathmodel)
        mkdir(outpathmodel)
    end

    outpathdefault = joinpath(outpathmodel, "default")
    if !isdir(outpathdefault)
        mkdir(outpathdefault)
    end

    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")
    if !isdir(outpath_hpo)
        mkdir(outpath_hpo)
    end

    path_to_use = outpath_hpo

    rs = []

    for item ∈ hpo_ranges[packagename][savename]
        if :values ∈ keys(item)
            push!(rs, range(mdl, item.hpname, values=item.values))
        else
            push!(rs, range(mdl, item.hpname, lower=item.lower, upper=item.upper))
        end
    end


    println("Performing hyperparameter optimization...")


    tuning = RandomSearch(rng=rng)

    if accelerate
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[mae, rsq, rms],  #first is used for the optimization but all are stored
            resampling=CV(nfolds=6, rng=rng), # this does ~ 85:15 split 6 times
            #resampling=Holdout(fraction_train=0.85, shuffle=true),
            acceleration=CPUThreads(),
            n=nmodels,
            cache=false,# define the total number of models to try
        )
    else
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[mae, rsq, rms],  #first is used for the optimization but all are stored
            resampling=CV(nfolds=6, rng=rng), # this does ~ 85:15 split 6 times
            # resampling=Holdout(fraction_train=0.85, shuffle=true),
            # acceleration=CPUThreads(),
            n=nmodels,
            cache=false,# define the total number of models to try
        )
    end

    # we are skipping repeats to save time...

    # bind to data and train:
    mach = machine(tuning_pipe, X, y; cache=false)

    println("Starting training...")
    fit!(mach, verbosity=0)

    println("...\tFinished training")
    println("Generating plots...")
    ŷ = MLJ.predict(mach, X)
    ŷtest = MLJ.predict(mach, Xtest)

    p1 = scatterresult(y, ŷ,
                      ytest, ŷtest;
                      xlabel="True $(target_long) [$(units)]",
                      ylabel="Predicted $(target_long) [$(units)]",
                      plot_title="Fit for $(longname)",)

    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).png"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).svg"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).pdf"))

    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          xlabel="True $(target_long) [$(units)]",
                          ylabel="Predicted $(target_long) [$(units)]",
                          title="Fit for $(longname)",)

    savefig(p2, joinpath(path_to_use, "qq__$(suffix).png"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).svg"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).pdf"))


    # save the model
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)

    open(joinpath(path_to_use, "$(savename)__hpo.txt"), "w") do f
        show(f,"text/plain", fitted_params(mach).best_model)
        println(f, "\n")
        println(f,"---------------------")
        show(f,"text/plain", fitted_params(mach).best_fitted_params)
        println(f,"\n")
        println(f,"---------------------")
        show(f,"text/plain", report(mach).best_history_entry)
        println(f,"\n")
        println(f,"---------------------")
        println(f, "r² train: $(rsq(ŷ, y))\tr² test:$(rsq(ŷtest, ytest))\tRMSE test: $(rmse(ŷtest, ytest))\tMAE test: $(mae(ŷtest, ytest))")
    end
end




# load in smart defaults
smart_defaults = JSON.parsefile("smart_hp_defaults.json")


function train_stack(y, X,
                   ytest, Xtest,
                   target_name, units, target_long,
                   outpath;
                   accelerate = true,
                   )

    longname = "Superlearner Stack"
    savename = "superlearner"
    suffix = "stack"

    # NNR = load NeuralNetworkRegressor pkg=MLJFlux
    # # nn = NNR(builder=MLJFlux.Short(n_hidden=50, σ=relu), epochs=25)
    # # ensemble_nn = EnsembleModel(atom=nn, n=20, bagging_fraction=bf)
    # nnr = NNR(builder=MLP((30, 30, 30, 30, 30), relu),
    #           batch_size = 32,
    #           optimiser=ADAM(0.01),
    #           rng=42,
    #           epochs=250,
    #           )

    nnr = NNR(builder=MLJFlux.MLP(hidden=(250,100,50), σ=NNlib.relu),
              batch_size = 200,
              optimiser=Flux.Optimise.ADAM(0.001),
              lambda = 0.0001,  # default regularization strength (I'm a bit confused by this as sk-learn only has alpha but that seems different here)
              rng=rng,
              epochs=200,
              )




    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpath_stack = joinpath(outpathtarget, "superlearner_stack")
    if !isdir(outpath_stack)
        mkdir(outpath_stack)
    end

    path_to_use = outpath_stack

    # go through each model and load the HPO optimized version.
    # -------- DTR -----------
    dtr = DTR()
    try
        path = joinpath(outpathtarget, "DecisionTreeRegressor", "hyperparameter_optimized")
        fpath = joinpath(path, "DecisionTreeRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(dtr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["DecisionTree"]["DecisionTreeRegressor"]
            println(p, "\t", val)
            setproperty!(dtr, Symbol(p), val)
        end
    end

    # -------- Ensemble of Trees -----------
    bf = 0.8
    edtr = EnsembleModel(model=dtr, n=100, bagging_fraction=bf)




    # -------- RFR -----------
    rfr = RFR()
    try
        path = joinpath(outpathtarget, "RandomForestRegressor", "hyperparameter_optimized")
        fpath = joinpath(path, "RandomForestRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(rfr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["DecisionTree"]["RandomForestRegressor"]
            println(p, "\t", val)
            setproperty!(rfr, Symbol(p), val)
        end
    end


    # -------- XGBR -----------
    xgbr = XGBR()
    try
        path = joinpath(outpathtarget, "XGBoostRegressor", "hyperparameter_optimized")
        fpath = joinpath(path, "XGBoostRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(xgbr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["XGBoost"]["XGBoostRegressor"]
            println(p, "\t", val)
            setproperty!(xgbr, Symbol(p), val)
        end
    end


    # -------- KNNR -----------
    knnr = KNNR()
    try
        path = joinpath(outpathtarget, "KNNRegressor", "hyperparameter_optimized")
        fpath = joinpath(path, "KNNRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(knnr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["NearestNeighborModels"]["KNNR"]
            println(p, "\t", val)
            setproperty!(knnr, Symbol(p), val)
        end
    end


    # -------- ETR -----------
    etr = ETR()
    try
        path = joinpath(outpathtarget, "EvoTreeRegressor", "hyperparameter_optimized")
        fpath = joinpath(path, "EvoTreeRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(etr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["EvoTrees"]["EvoTreeRegressor"]
            println(p, "\t", val)
            setproperty!(etr, Symbol(p), val)
        end
    end


    # -------- LGBR -----------
    lgbr = LGBR()
    try
        path = joinpath(outpathtarget, "LGBMRegreesor", "hyperparameter_optimized")
        fpath = joinpath(path, "LGBMRegressor__hpo.jls")
        mach = machine(fpath)

        ps = params(fitted_params(mach).best_model)
        for (p, val) ∈ zip(keys(ps), ps)
            println(p, "\t", val)
            setproperty!(lgbr, Symbol(p), val)
        end
    catch e
        println("couldnt find hpo results. Loading smart defaults instead")
        for (p, val) ∈ smart_defaults["LightGBM"]["LGBMRegressor"]
            println(p, "\t", val)
            setproperty!(lgbr, Symbol(p), val)
        end
    end



    if accelerate
        stack = Stack(;
                      #                      metalearner=LR(),
                      metalearner=RR(),
                      dtr=dtr,
                      rfr=rfr,
                      edtr=edtr,
                      xgbr=xgbr,
                      knnr=knnr,
                      etr=etr,
                      lgbr=lgbr,
                      nnr=nnr,
                      resampling=CV(nfolds=6, rng=rng),
                      acceleration=CPUThreads(),
                      cache=false,
                      )
    else
        stack = Stack(;
                      #                      metalearner=LR(),
                      metalearner=RR(),
                      dtr=dtr,
                      rfr=rfr,
                      edtr=edtr,
                      xgbr=xgbr,
                      knnr=knnr,
                      etr=etr,
                      lgbr=lgbr,
                      nnr=nnr,
                      resampling=CV(nfolds=6, rng=rng),
                      cache=false,
                      )
    end

    # we are skipping repeats to save time...

    # bind to data and train:
    mach = machine(stack, X, y; cache=false)

    println("Starting training...")
    fit!(mach, verbosity=0)

    println("...\tFinished training")
    println("Generating plots...")
    ŷ = MLJ.predict(mach, X)
    ŷtest = MLJ.predict(mach, Xtest)

    p1 = scatterresult(y, ŷ,
                      ytest, ŷtest;
                      xlabel="True $(target_long) [$(units)]",
                      ylabel="Predicted $(target_long) [$(units)]",
                      plot_title="Fit for $(longname)",)

    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).png"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).svg"))
    savefig(p1, joinpath(path_to_use, "scatterplt__$(suffix).pdf"))

    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          xlabel="True $(target_long) [$(units)]",
                          ylabel="Predicted $(target_long) [$(units)]",
                          title="Fit for $(longname)",)

    savefig(p2, joinpath(path_to_use, "qq__$(suffix).png"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).svg"))
    savefig(p2, joinpath(path_to_use, "qq__$(suffix).pdf"))


    # save the model
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)

    open(joinpath(path_to_use, "$(savename)__$(suffix).txt"), "w") do f
        println(f,"\n")
        println(f,"---------------------")
        println(f, "r² train: $(rsq(ŷ, y))\tr² test:$(rsq(ŷtest, ytest))\tRMSE test: $(rmse(ŷtest, ytest))\tMAE test: $(mae(ŷtest, ytest))")
    end
end
