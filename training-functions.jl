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
                  # "XGBoost" => Dict("XGBoostRegressor" => [
                  #                                          (hpname=:num_round, values=[50,75,100,125,150]),
                  #                                          (hpname=:eta, lower=0.01, upper=0.5),
                  #                                          #(hpname=:gamma, lower=0, upper=100),
                  #                                          (hpname=:max_depth, values=[3,4,5,6,7,8]),
                  #                                          #(hpname=:max_delta_step, lower=0.0, upper=10.0),
                  #                                          (hpname=:min_child_weight, values=[1,10,100]),
                  #                                          (hpname=:subsample, lower=0.5, upper=1.0),
                  #                                          (hpname=:lambda, lower=0.1, upper=5.0),
                  #                                          (hpname=:alpha, lower=0.0, upper=10.0),
                  #                                          ],
                  #                   ),
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
                  # "EvoTrees" => Dict("EvoTreeRegressor" => [(hpname=:nrounds,values=[50,75,100,125,150, 175, 200]),
                  #                                           (hpname=:nbins, lower=2, upper=255),
                  #                                           (hpname=:eta, lower=0.001, upper=0.5),
                  #                                           (hpname=:max_depth, values=[3,4,5,6,7,8,9,10]),
                  #                                           (hpname=:rowsample, lower=0.5, upper=1.0),
                  #                                           #(hpname=:L2, lower=0.0, upper=100.0),
                  #                                           #(hpname=:lambda, lower=0.1, upper=5.0), 
                  #                                           #(hpname=:alpha, lower=0.0, upper=1.0),
                  #                                           ],
                  #                    ),
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
                  # "LightGBM" => Dict("LGBMRegressor" => [(hpname=:num_iterations, lower=2, upper=100),
                  #                                        (hpname=:learning_rate, lower=0.01, upper=0.3),
                  #                                        (hpname=:max_depth, values=[3,4,5,6,7,8,9,10]),
                  #                                        (hpname=:bagging_fraction, lower=0.65, upper=1.0),
                  #                                        ]),
                  "LightGBM" => Dict("LGBMRegressor" => [(hpname=:num_iterations, lower=5, upper=100),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.3),
                                                         (hpname=:max_depth, lower=3, upper=10),
                                                         (hpname=:bagging_fraction, lower=0.5, upper=1.0),
                                                         ]),
                  "CatBoost" => Dict("CatBoostRegressor" => [(hpname=:iterations, lower=100, upper=1500),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.1),
                                                         (hpname=:max_depth, values=[3,4,5,6,7,8,9,10]),
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


    @info "\tSetting up save paths"

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
        @info "\t$(longname) does not have parameter :rng"
    end

    # verify scitype is satisfied
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)



    conf_model = conformal_model(mdl; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

    # fit the machine
    @info "\tTraining model"
    mach = machine(conf_model, X, y)
    fit!(mach)

    reports_feature_importances(conf_model)

    # generate predictions
    @info "\tGenerating predictions"
    ŷtrain = ConformalPrediction.predict(mach, X);
    ŷtest = ConformalPrediction.predict(mach, Xtest);

    # compute coverage on test set
    cov = emp_coverage(ŷtest, ytest);
    @info "\tEmpirical coverage on test set: $(round(cov, digits=3))"


    # convert to predictions + uncertainties
    ϵtrain = [abs(f[2] - f[1]) / 2 for f ∈ ŷtrain];
    ŷtrain = mean.(ŷtrain);

    ϵtest = [abs(f[2] - f[1]) / 2 for f ∈ ŷtest];
    ŷtest = mean.(ŷtest);


    # generate scatterplot
    @info "\tCreating Plots"
    fig = scatter_results(y, ŷtrain, ϵtrain, ytest, ŷtest, ϵtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    # save the model println("\tSaving the model")
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)


    res_dict = Dict()
    res_dict["rsq_train"] = rsq(ŷtrain, y)
    res_dict["rsq_test"] = rsq(ŷtest, ytest)
    res_dict["rmse_train"] = rmse(ŷtrain, y)
    res_dict["rmse_test"] = rmse(ŷtest, ytest)
    res_dict["mae_train"] = mae(ŷtrain, y)
    res_dict["mae_test"] = mae(ŷtest, ytest)
    res_dict["cov"] = cov
    open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
        JSON.print(f, res_dict)
    end



    # 2. Compute feature importances
    if reports_feature_importances(mdl)
        n_features = 25

        @info "\tComputing feature importances"
        rpt = report(mach);

        fi_pairs = feature_importances(mach.model.model, mach.fitresult, rpt);
        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
        sort!(fi_df, :rel_importance; rev=true)

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

        # do the same for ONLY reflectances
        ref_vals = Symbol.(["R_" * lpad(i, 3, "0") for i in 1:462])
        fi_pairs_ref = [p for p in fi_pairs if (p.first in ref_vals)]
        fi_df_ref = DataFrame()
        fi_df_ref.feature_name = [x[1] for x ∈ fi_pairs_ref]
        fi_df_ref.rel_importance = [x[2] for x ∈ fi_pairs_ref]
        fi_df_ref.rel_importance .= fi_df_ref.rel_importance ./ maximum(fi_df_ref.rel_importance)
        sort!(fi_df_ref, :rel_importance; rev=true)
        
        rel_importance = fi_df_ref.rel_importance[1:n_features]
        var_names = [name_replacements[s] for s ∈ String.(fi_df_ref.feature_name[1:n_features])]


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

        save(joinpath(outpath_featuresreduced, "importance_ranking_refs_only__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "importance_ranking_refs_only__$(suffix).pdf"), fig)


        # plot the correlation of Ref w/ target as function of wavelength

        # grab most important wavelengths
        # λ_important = Float64[]
        # n_important_to_use = 10
        # for name in String.(fi_df_ref.feature_name[1:n_important_to_use])
        #     idx = parse(Int, split(name, "_")[2])
        #     push!(λ_important, wavelengths[idx])
        # end

        # println(λ_important)

        # # compute pearson correlation
        # ref_names = ["R_" * lpad(i, 3, "0") for i in 1:462]
        # cvals = cor(Matrix(X[:, ref_names]), y)[:,1]

        # fig = Figure();

        # ylabel_title = "Correlation with $(target_long)"
        # if length(ylabel_title) > 45
        #     ylabel_title = "Correlation with\n$(target_long)"
        # end

        # ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel=ylabel_title)

        # hlines!(ax, [0.0], color=(:black, 0.25), linewidth=2)
        # l = lines!(ax, wavelengths, cvals, linewidth=3, color=:black)
        # vl = vlines!(λ_important, color=:gray, linewidth=2, linestyle=:dashdot)
        # axislegend(ax, [l, vl], ["Correlation", "Important Wavelengths"]; framevisible=false)
        # xlims!(ax, wavelengths[1], wavelengths[end])
        # fig

        # save(joinpath(outpath_featuresreduced, "correlation-with-$(target_name)__$(suffix).png"), fig)
        # save(joinpath(outpath_featuresreduced, "correlation-with-$(target_name)__$(suffix).pdf"), fig)




        # now retrain the model with a limited number of features
        N_features = 100

        fi_n = @view fi_df[1:N_features, :]

        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_df.feature_name)),
            model=mdl,
        )


        conf_model = conformal_model(pipe; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

        mach_fi = machine(conf_model, X, y)
        fit!(mach_fi)


        @info "\tGenerating predictions"
        ŷtrain = ConformalPrediction.predict(mach_fi, X);
        ŷtest = ConformalPrediction.predict(mach_fi, Xtest);

        # compute coverage on test set
        cov = emp_coverage(ŷtest, ytest);
        @info "\tEmpirical coverage on test set: $(round(cov, digits=3))"
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

        res_dict = Dict()
        res_dict["rsq_train"] = rsq(ŷtrain, y)
        res_dict["rsq_test"] = rsq(ŷtest, ytest)
        res_dict["rmse_train"] = rmse(ŷtrain, y)
        res_dict["rmse_test"] = rmse(ŷtest, ytest)
        res_dict["mae_train"] = mae(ŷtrain, y)
        res_dict["mae_test"] = mae(ŷtest, ytest)
        res_dict["cov"] = cov
        open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
            JSON.print(f, res_dict)
        end
    end

    # return rsq(ŷtrain, y), rsq(ŷtest, ytest), rmse(ŷtrain, y), rmse(ŷtest, ytest), cov
    nothing
end






function train_hpo(
    X, y,
    Xtest, ytest,
    longname, savename, packagename,
    target_name, units, target_long,
    mdl,
    outpath;
    nmodels=200,
    accelerate=false,
    rng=Xoshiro(42),
    unc_method=:simple_inductive,
    train_ratio=0.91,
    unc_coverage=0.9,
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


    Xhpo = vcat(X, Xtest)
    yhpo = vcat(y, ytest)

    rows_train = 1:nrow(X)
    rows_test = (nrow(X)+1):nrow(Xhpo)

    # search for hyperparameters without doing conformal prediction
    tuning = RandomSearch(rng=rng)
    if accelerate
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[rmse, mae, rsq],
            resampling=CV(nfolds=10, rng=rng),
            #resampling=[(rows_train, rows_test),],
            acceleration=CPUThreads(),
            n=nmodels,
            cache=false,
        )
    else
        tuning_pipe = TunedModel(
            model=mdl,
            range=rs,
            tuning=tuning,
            measures=[rmse, mae, rsq],
            resampling=CV(nfolds=10, rng=rng),
            n=nmodels,
            cache=false,
        )
    end


    mach = machine(tuning_pipe, X, y; cache=false)

    @info "\tStarting training..."
    fit!(mach) #, verbosity=0)

    @info "\t...\tFinished training"

    open(joinpath(path_to_use, "$(savename)__hpo.txt"), "w") do f
        show(f,"text/plain", fitted_params(mach).best_model)
        println(f, "\n")
        println(f,"---------------------")
        show(f,"text/plain", fitted_params(mach).best_fitted_params)
        println(f,"\n")
        println(f,"---------------------")
        show(f,"text/plain", report(mach).best_history_entry)
        println(f,"\n")
    end


    @info "\tSaving params to json file"
    fitted_ps = fitted_params(mach).best_model
    params_dict = Dict()
    for hp in hpo_ranges[packagename][savename]
        val = getproperty(fitted_ps, hp.hpname)
        name = hp.hpname
        params_dict[name] = val
    end

    open(joinpath(path_to_use, "hp_defaults.json"), "w") do f
        JSON.print(f, params_dict)
    end



    # now we want to train and evaluate a final model w/ conformal prediction
    mdl_final = mdl
    for (key, val) in params_dict
        setproperty!(mdl_final, Symbol(key), val)
    end

    conf_model = conformal_model(mdl_final; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

    # fit the machine
    @info "\tTraining final hpo model"
    mach = machine(conf_model, X, y)
    fit!(mach)

    # generate predictions
    @info "\tGenerating predictions"
    ŷtrain = ConformalPrediction.predict(mach, X);
    ŷtest = ConformalPrediction.predict(mach, Xtest);

    # compute coverage on test set
    cov = emp_coverage(ŷtest, ytest);
    @info "\tEmpirical coverage on test set: $(round(cov, digits=3))"


    # convert to predictions + uncertainties
    ϵtrain = [abs(f[2] - f[1]) / 2 for f ∈ ŷtrain];
    ŷtrain = mean.(ŷtrain);

    ϵtest = [abs(f[2] - f[1]) / 2 for f ∈ ŷtest];
    ŷtest = mean.(ŷtest);


    # generate scatterplot
    @info "\tCreating Plots"
    fig = scatter_results(y, ŷtrain, ϵtrain, ytest, ŷtest, ϵtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    # save the model println("\tSaving the model")
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)


    res_dict = Dict()
    res_dict["rsq_train"] = rsq(ŷtrain, y)
    res_dict["rsq_test"] = rsq(ŷtest, ytest)
    res_dict["rmse_train"] = rmse(ŷtrain, y)
    res_dict["rmse_test"] = rmse(ŷtest, ytest)
    res_dict["mae_train"] = mae(ŷtrain, y)
    res_dict["mae_test"] = mae(ŷtest, ytest)
    res_dict["cov"] = cov
    open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
        JSON.print(f, res_dict)
    end

end





cols_to_use = ["R_" * lpad(i, 3, "0") for i in 1:462]
cols_to_standardize = ["roll", "pitch", "heading", "view_angle", "solar_azimuth", "solar_elevation", "solar_zenith"]
cols_to_use = vcat(cols_to_use, cols_to_standardize)


function train_stack(
    X, y,
    Xtest, ytest,
    target_name, units, target_long,
    outpath;
    accelerate=false,
    unc_method=:simple_inductive,
    train_ratio=0.91,
    unc_coverage=0.9,
    rng=Xoshiro(42),
    )

    longname = "Model Stack"
    savename = "superlearner"
    suffix = "stack"

    @info "\tSetting up save paths"

    outpathdefault = joinpath(outpath, savename, "default")
    outpath_featuresreduced = joinpath(outpath, savename, "important_only")
    outpath_hpo = joinpath(outpath, savename, "hyperparameter_optimized")
    outpath_stack = joinpath(outpath, savename, "superlearner_stack")

    for path ∈ [outpathdefault, outpath_featuresreduced, outpath_hpo, outpath_stack]
        if !isdir(path)
            mkpath(path)
        end
    end

    path_to_use = outpath_stack


    # 1. Train model
    @info "\tSetting model random seed"

    # go through each model and load the hpo optimized version where possible


    # ------------ NNR ----------------------
    nnr_mod = NNR(builder=MLJFlux.MLP(hidden=(50,50,50,50), σ=Flux.relu),
                  batch_size=256,
                  optimiser=Flux.Optimise.ADAM(),
                  lambda = 0.0001,
                  rng=42,
                  epochs=300,
                  )


    nnr  = Pipeline(
        selector=FeatureSelector(features=Symbol.(cols_to_use)),
        stand=Standardizer(features=Symbol.(cols_to_standardize)),
        mdl=nnr_mod
    )


    # -----------Bag of Trees --------------

    dtr = DTR()
    try
        hp_path = joinpath(outpath, "DecisionTreeRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(dtr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for DTR"
    end

    bf = 0.8
    bdtr = EnsembleModel(model=dtr, n=100, bagging_fraction=bf)


    # --------- RFR -----------------------
    rfr = RFR()
    try
        hp_path = joinpath(outpath, "RandomForestRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(rfr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for RFR"
    end


    # -------- XGBR -----------
    xgbr = XGBR()
    try
        hp_path = joinpath(outpath, "XGBoostRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(xgbr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for XGBR"
    end


    # -------- KNNR -----------
    knnr = KNNR()
    try
        hp_path = joinpath(outpath, "KNNRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(knnr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for KNNR"
    end


    # -------- ETR -----------
    etr = ETR()
    try
        hp_path = joinpath(outpath, "EvoTreeRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(etr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for ETR"
    end


    # -------- LGBR -----------
    lgbr = LGBR()
    try
        hp_path = joinpath(outpath, "LGBMRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(lgbr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for KNNR"
    end

    # -------- CatR -----------
    catr = CatR()
    try
        hp_path = joinpath(outpath, "CatBoostRegressor", "hyperparameter_optimized", "hp_defaults.json")
        hp_defaults = JSON.parsefile(hp_path)
        for (key, val) in hp_defaults
            setproperty!(catr, Symbol(key), val)
        end
    catch e
        @info "\tCouldnt set hp values for KNNR"
    end


    if accelerate
        stack = Stack(;
                      #                      metalearner=LR(),
                      metalearner=RR(),
                      bdtr=bdtr,
                      rfr=rfr,
                      xgbr=xgbr,
                      knnr=knnr,
                      etr=etr,
                      lgbr=lgbr,
                      nnr=nnr,
                      catr=catr,
                      resampling=CV(nfolds=10, rng=rng),
                      acceleration=CPUThreads(),
                      cache=false,
                      )
    else
        stack = Stack(;
                      #                      metalearner=LR(),
                      metalearner=DTR(),
                      bdtr=bdtr,
                      rfr=rfr,
                      xgbr=xgbr,
                      knnr=knnr,
                      etr=etr,
                      lgbr=lgbr,
                      nnr=nnr,
                      catr=catr,
                      resampling=CV(nfolds=10, rng=rng),
                      cache=false,
                      )
    end



    conf_model = conformal_model(stack; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

    # fit the machine
    mach = machine(conf_model, X, y; cache=false)


    @info "\tStarting training..."
    fit!(mach) #, verbosity=0)

    @info "\t...\tFinished training"

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




    @info "\tGenerating plots..."

    # generate scatterplot
    fig = scatter_results(y, ŷtrain, ϵtrain, ytest, ŷtest, ϵtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    @info "\tSaving hpo model..."

    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)


    res_dict = Dict()
    res_dict["rsq_train"] = rsq(ŷtrain, y)
    res_dict["rsq_test"] = rsq(ŷtest, ytest)
    res_dict["rmse_train"] = rmse(ŷtrain, y)
    res_dict["rmse_test"] = rmse(ŷtest, ytest)
    res_dict["cov"] = cov
    open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
        JSON.print(f, res_dict)
    end
    nothing
end

