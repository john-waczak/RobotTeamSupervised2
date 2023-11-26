#import MLJModelInterface as MMI
using JSON

# define dictionary with default hyperparameter values used for tuning for each model
# use a dict of dicts structure

# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook

hpo_ranges = Dict("DecisionTree" => Dict("DecisionTreeRegressor" => [(hpname=:min_samples_leaf, lower=2, upper=100),
                                                                      (hpname=:max_depth, values=[-1, 2, 3, 5, 10, 20]),
                                                                      (hpname=:post_prune, values=[false, true])
                                                                      ],
                                         "RandomForestRegressor" => [
                                                                      (hpname=:min_samples_leaf, lower=2, upper=100),
                                                                      (hpname=:max_depth, values=[-1, 2, 3, 5, 10, 20]),
                                                                      (hpname=:n_subfeatures, values=[-1,0]),
                                                                      (hpname=:n_trees, values=[10, 25, 50, 75, 100, 125, 150]),
                                                                      (hpname=:sampling_fraction, lower=0.65, upper=0.9)
                                                                      ],
                                          ),
                  "XGBoost" => Dict("XGBoostRegressor" => [(hpname=:num_round, values=[50,75,100,125,150]),
                                                           (hpname=:eta, lower=0.01, upper=0.5),
                                                           (hpname=:max_depth, values=[3,4,5,6,7,8,9]),
                                                           (hpname=:lambda, lower=0.1, upper=5.0),  # L2 regularization. Higher makes model more conservative
                                                           (hpname=:alpha, lower=0.0, upper=1.0), # L1 regularization. Higher makes model more sparse
                                                           ],
                                    ),
                  "EvoTrees" => Dict("EvoTreeRegressor" => [(hpname=:nrounds, lower=2, upper=100),
                                                            (hpname=:eta, lower=0.01, upper=0.2),
                                                            (hpname=:max_depth, values=[3,4,5,6,7,8,9]),
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
                  "LightGBM" => Dict("LGBMRegressor" => [(hpname=:num_iterations, lower=2, upper=100),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.3),
                                                         (hpname=:max_depth, values=[3,4,5,6,7,8,9]),
                                                         (hpname=:bagging_fraction, lower=0.65, upper=1.0),
                                                         ]),
                  "CatBoost" => Dict("CatBoostRegressor" => [(hpname=:iterations, lower=100, upper=1500),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.1),
                                                         (hpname=:max_depth, lower=3, upper=9),
                                                         ])
                  )




function train_basic(
    X, y,
    Xtest, ytest,
    longname, savename, mdl,
    target_name, units, target_long,
    outpath;
    suffix="",
    unc_method=:simple_inductive,
    train_ratio=0.91,
    unc_coverage=0.9,
    rng=Xoshiro(42),
    N_features=100,
    )


    println("\tSetting up save paths")

    outpathdefault = joinpath(outpath, savename, "default")
    outpath_featuresreduced = joinpath(outpath, savename, "important_only")

    path_to_use = outpathdefault

    for path ∈ [outpathdefault, outpath_featuresreduced]
        if !isdir(path)
            mkpath(path)
        end
    end


    # 1. Train model

    try
        mdl.rng = rng  # if there is an rng parameter, set it for reproducability
    catch
        println("\t$(longname) does not have parameter :rng")
    end

    # verify scitype is satisfied
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)



    conf_model = conformal_model(mdl; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

    # fit the machine
    println("\tTraining model")
    mach = machine(conf_model, X, y)
    fit!(mach)

    reports_feature_importances(conf_model)

    # generate predictions
    println("\tGenerating predictions")
    ŷtrain = ConformalPrediction.predict(mach, X);
    ŷtest = ConformalPrediction.predict(mach, Xtest);

    # compute coverage on test set
    cov = emp_coverage(ŷtest, ytest);
    println("\tEmpirical coverage on test set: $(round(cov, digits=3))")


    # convert to predictions + uncertainties
    ϵtrain = [abs(f[2] - f[1]) / 2 for f ∈ ŷtrain];
    ŷtrain = mean.(ŷtrain);

    ϵtest = [abs(f[2] - f[1]) / 2 for f ∈ ŷtest];
    ŷtest = mean.(ŷtest);


    # generate scatterplot
    println("\tCreating Plots")
    fig = scatter_results(y, ŷtrain, ϵtrain, ytest, ŷtest, ϵtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    # save the model println("\tSaving the model")
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)



    # 2. Compute feature importances
    if reports_feature_importances(mdl)
        n_features = 25

        println("\tComputing feature importances")

        rpt = report(mach);
        fi_pairs = feature_importances(mach.model.model, mach.fitresult, rpt);
        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)

        CSV.write(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).csv"), fi_df)

        rel_importance = fi_df.rel_importance[1:n_features]
        var_names = [name_replacements[s] for s ∈ String.(fi_df.feature_name[1:n_features])]


        var_colors = cgrad([mints_colors[2], mints_colors[1]], n_features)[1:n_features]


        fig = Figure(; resolution=(1000, 1000))
        ax = Axis(fig[1, 1],
            yticks=(1:n_features, var_names[end:-1:1]),
            xlabel="Normalized Feature Importance",
            title="$(target_long)",
            yminorgridvisible=false,
        )

        b = barplot!(ax,
            1:n_features,
            rel_importance[end:-1:1],
            direction=:x,
            color=var_colors
        )

        xlims!(ax, -0.01, 1.025)
        ylims!(ax, 0.5, n_features + 0.5)

        save(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).pdf"), fig)



        # now retrain the model with a limited number of features
        N_features = 100

        fi_n = @view fi_df[1:N_features, :]

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
            model=mdl,
        )


        conf_model = conformal_model(pipe; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

        mach_fi = machine(conf_model, X, y)
        fit!(mach_fi)


        println("\tGenerating predictions")
        ŷtrain = ConformalPrediction.predict(mach_fi, X);
        ŷtest = ConformalPrediction.predict(mach_fi, Xtest);

        # compute coverage on test set
        cov = emp_coverage(ŷtest, ytest);
        println("\tEmpirical coverage on test set: $(round(cov, digits=3))")
        # convert to predictions + uncertainties
        ϵtrain = [abs(f[2] - f[1]) / 2 for f ∈ ŷtrain];
        ŷtrain = mean.(ŷtrain);

        ϵtest = [abs(f[2] - f[1]) / 2 for f ∈ ŷtest]
        ŷtest = mean.(ŷtest)

        fig = scatter_results(y, ŷtrain, ϵtrain, ytest, ŷtest, ϵtest, "$(target_long) ($(units))")

        save(joinpath(outpath_featuresreduced, "scatterplot__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "scatterplot__$(suffix).pdf"), fig)


        # generate quantile-quantile plot
        fig = quantile_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
        save(joinpath(outpath_featuresreduced, "qq__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "qq__$(suffix).pdf"), fig)
    end

    return rsq(ŷtrain, y), rsq(ŷtest, ytest), rmse(ŷtrain, y), rmse(ŷtest, ytest), cov
end






function train_hpo(
    X, y,
    Xtest, ytest,
    longname, savename, packagename,
    target_name, units, target_long,
    mdl,
    outpath;
    nmodels=20,
    accelerate=true,
    rng=Xoshiro(42),
    )

    suffix = "hpo"

    @info "\tSetting up save paths"

    outpathdefault = joinpath(outpath, savename, "default")
    outpath_featuresreduced = joinpath(outpath, savename, "important_only")
    outpath_hpo = joinpath(outpath, savename, "hyperparameter_optimized")

    for path ∈ [outpathdefault, outpath_featuresreduced, outpath_hpo]
        if !isdir(path)
            mkpath(path)
        end
    end

    path_to_use = outpath_hpo

    # 1. Train model
    @info "\tSetting model random seed"


    if :rng in fieldnames(typeof(mdl))
        mdl.rng = rng
    end


    @info "\tVerifying scitype is satisfied"
    # verify scitype is satisfied
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)



    # set up hyperparameter rangese
    rs = []

    # Either add allowed values or ranges for hyperparameters
    for item in hpo_ranges[packagename][savename]
        if :values ∈ keys(item)
            push!(rs, range(mdl, item.hpname, values=item.values))
        else
            push!(rs, range(mdl, item.hpname, lower=item.lower, upper=item.upper))
        end
    end


    @info "\tPerforming hyperparameter optimization..."


    # search for hyperparameters without doing conformal prediction
    tuning = RandomSearch(rng=rng)
    if accelerate
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[mae, rsq, rms],
            resampling=CV(nfolds=10, rng=rng),
            acceleration=CPUThreads(),
            n=nmodels,
            cache=false,
        )
    else
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[mae, rsq, rms],
            resampling=CV(nfolds=10, rng=rng),
            n=nmodels,
            cache=false,
        )
    end


    mach = machine(tuning_pipe, X, y; cache=false)

    @info "\tStarting training..."
    fit!(mach) #, verbosity=0)

    @info "\t...\tFinished training"


    @info "\tGenerating plots..."
    ŷ = MLJ.predict(mach, X)
    ŷtest = MLJ.predict(mach, Xtest)


    # generate scatterplot
    fig = scatter_results(y, ŷ, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(y, ŷ, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    @info "\tSaving hpo model..."

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


    @info "\tSaving params to json file"
    fitted_ps = fitted_params(mach).best_model
    params_dict = Dict()
    for hp in hpo_ranges[packagename][savename]
        val = getproperty(fitted_ps, hp.hpname)
        name = hp.hpname
        params_dict[name] = val
    end

    println(params_dict)
    println(joinpath(path_to_use, "hp_defaults.json"))

    open(joinpath(path_to_use, "hp_defaults.json"), "w") do f
        JSON.print(f, params_dict)
    end

end

