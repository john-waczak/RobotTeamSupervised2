#import MLJModelInterface as MMI
using JSON
using Statistics
# define dictionary with default hyperparameter values used for tuning for each model
# use a dict of dicts structure

# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
hpo_ranges = Dict("DecisionTree" => Dict("DecisionTreeRegressor" => [(hpname=:min_samples_leaf, lower=2, upper=100),
                                                                      (hpname=:max_depth, values=[-1, 2, 3, 5, 10, 20]),
                                                                      (hpname=:post_prune, values=[false, true])
                                                                      ],
                                         "RandomForestRegressor" => [
                                                                      (hpname=:min_samples_leaf, lower=1, upper=100),
                                                                      #(hpname=:max_depth, values=[-1, 3, 5, 7]),
                                                                      #(hpname=:n_subfeatures, lower=10, upper=100,),
                                                                      #(hpname=:n_trees, lower=50, upper=200),
                                                                      (hpname=:sampling_fraction, lower=0.8, upper=1.0)
                                                                      ],
                                          ),
                 "XGBoost" => Dict("XGBoostRegressor" => [(hpname=:num_round, lower=50, upper=100),
                                                          (hpname=:eta, lower=0.01, upper=0.5),
                                                          (hpname=:max_depth, lower=3, upper=6),
                                                          (hpname=:subsample, lower=0.65, upper=1.0),
                                                          (hpname=:colsample_bytree, lower=0.65, upper=1.0),
                                                          (hpname=:lambda, lower=0.0, upper=100.0),  # L2 regularization. Higher makes model more conservative
                                                          (hpname=:alpha, lower=0.0, upper=100.0), # L1 regularization. Higher makes model more sparse
                                                          ],
                                   ),
                  "EvoTrees" => Dict("EvoTreeRegressor" => [#(hpname=:nrounds,lower=50, upper=150),
                                                            #(hpname=:nbins, lower=64, upper=255),
                                                            (hpname=:eta, lower=0.2, upper=0.4),
                                                            #(hpname=:max_depth, lower=3, upper=8),
                                                            #(hpname=:rowsample, lower=0.65, upper=1.0),
                                                            #(hpname=:colsample, lower=0.65, upper=1.0),
							                                              #(hpname=:L2, lower=0.0, upper=1000.0),
                                                            #(hpname=:lambda, lower=0.0, upper=1000.0),
                                                            #(hpname=:alpha, lower=0.0, upper=1000.0),
                                                            (hpname=:lambda, lower=0.1, upper=10.0),
                                                            (hpname=:alpha, lower=0.0, upper=10.0),
                                                            ],
                                     ),
                 "NearestNeighborModels" => Dict("KNNRegressor" => [(hpname=:K, lower=1, upper=50),
                                                                     (hpname=:leafsize, lower=1, upper=50),
                                                                     ],
                                                  ),
                  "MLJFlux" => Dict("NeuralNetworkRegressor" =>[],
                                    ),
                  "LightGBM" => Dict("LGBMRegressor" => [(hpname=:num_iterations, lower=5, upper=100),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.5),
                                                         (hpname=:max_depth, lower=3, upper=6),
                                                         (hpname=:bagging_fraction, lower=0.5, upper=1.0),
                                                         ]),
                  "CatBoost" => Dict("CatBoostRegressor" => [(hpname=:iterations, lower=100, upper=1500),
                                                         (hpname=:learning_rate, lower=0.01, upper=0.1),
                                                         (hpname=:max_depth, values=[3,4,5,6,7,8,9,10]),
                                                         ])
                  )





cor_coef(yhat, y) = Statistics.cor(yhat, y)


function train_folds(
    X, y,
    idx_train, idx_test,
    longname, savename, mdl,
    target_name, units, target_long,
    outpath;
    suffix="",
    rng=Xoshiro(42),
    nfolds=10,
    )

    @info "\tSetting up save paths"

    outpathdefault = joinpath(outpath, savename, "default")
    path_to_use = outpathdefault
    if !ispath(path_to_use)
        mkpath(path_to_use)
    end


    try
        mdl.rng = rng  # if there is an rng parameter, set it for reproducability
    catch
        @info "\t$(longname) does not have parameter :rng"
    end

    # verify scitype is satisfied
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)


    # fit the machine
    Xtrain = @view X[idx_train,:]
    ytrain = @view y[idx_train]

    Xtest = @view X[idx_test,:]
    ytest = @view y[idx_test]

    mach = machine(mdl, Xtrain, ytrain)

    @info "\tTraining model on full training set..."
    fit!(mach)

    yhat_train = MLJ.predict(mach, Xtrain)
    yhat_test = MLJ.predict(mach, Xtest)


    # generate plots:
    @info "\tGenerating plots"
    fig = scatter_results(
        ytrain,
        yhat_train,
        ytest,
        yhat_test,
        "$(target_long) ($(units))"
    )

    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    fig = quantile_results(
        ytrain,
        yhat_train,
        ytest,
        yhat_test,
        "$(target_long) ($(units))"
    )
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)

    @info "\tSaving model"
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)


    # 2. Compute feature importances
    if reports_feature_importances(mdl)
        n_features = 25

        @info "\tComputing feature importances"
        rpt = report(mach);

        fi_pairs = feature_importances(mach) #.model, mach.fitresult, rpt);


        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
        sort!(fi_df, :rel_importance; rev=true)

        CSV.write(joinpath(path_to_use, "importance_ranking__$(suffix).csv"), fi_df)

        rel_importance = fi_df.rel_importance[1:n_features]
        var_names = [name_replacements[s] for s ∈ String.(fi_df.feature_name[1:n_features])]


        var_colors = cgrad([mints_colors[2], mints_colors[1]], n_features)[1:n_features]

        fig = Figure(; resolution=(1000, 1000))
        ax = Axis(
            fig[1, 1],
            yticks=(1:n_features, var_names[end:-1:1]),
            xlabel="Normalized Feature Importance",
            title="$(target_long)",
            yminorgridvisible=false,
            xscale=log10,
        )

        b = barplot!(ax,
            1:n_features,
            rel_importance[end:-1:1] .+ eps(1.0),
            direction=:x,
            color=var_colors
        )

        #xlims!(ax,-0.01, 1.025)
        ylims!(ax, 0.5, n_features + 0.5)

        save(joinpath(path_to_use, "importance_ranking__$(suffix).png"), fig)
        save(joinpath(path_to_use, "importance_ranking__$(suffix).pdf"), fig)


        # run occam
        @info "\tRunning Occam Selection..."
        N_features_to_use = [1,5,10,15,25,50,100]
        r2_vals = Float64[]
        rmse_vals = Float64[]

        for N_features in N_features_to_use
            @info "\tEvaluating models for $(N_features) features..."
            fi_n = @view fi_df[1:N_features, :]

            try
                pipe = Pipeline(
                    selector=FeatureSelector(features=Symbol.(fi_n.feature_name)),
                    model=mdl,
                )


                mach = machine(pipe, Xtrain, ytrain)
                cv = CV(nfolds=6, rng=rng)

                e = evaluate!(
                    mach,
                    resampling=cv,
                    measures=[rsq, rmse, mae, cor_coef]
                )

                push!(r2_vals, mean(e.per_fold[1]))
                push!(rmse_vals, mean(e.per_fold[2]))

            catch e
                println(e)
            end
        end

        # now let's plot the error as a function of number of features and pick the best
        fig = Figure();
        ax = Axis(fig[1,1], xlabel="N features", ylabel="R² for $(target_long)")
        ax2 = Axis(fig[1,1], ylabel="RMSE for $(target_long)", yaxisposition = :right)
        hidespines!(ax2)
        hidexdecorations!(ax2)

        l1 = lines!(ax, N_features_to_use, r2_vals, linewidth=3, color=mints_colors[1])
        l2 = lines!(ax2, N_features_to_use, rmse_vals, linewidth=3, color=mints_colors[2], linestyle=:dash)

        save(joinpath(path_to_use, "feature-dependence__$(suffix).png"), fig)
        save(joinpath(path_to_use, "feature-dependence__$(suffix).pdf"), fig)

        # train feature reduced model
        res_dict = Dict()
        res_dict["n_best"] = N_features_to_use[argmax(r2_vals)]

        N_final = 25

        @info "\tTraining model with $(N_final) features..."

        fi_n = @view fi_df[1:N_final, :]
        fi_occam = Symbol.(fi_n.feature_name)
        Xtrain = @view X[idx_train, fi_occam]
        Xtest = @view X[idx_test, fi_occam]

        mach_occam = machine(mdl, Xtrain, ytrain)
        cv = CV(nfolds=nfolds, rng=rng)

        e = evaluate!(
            mach_occam,
            resampling=cv,
            measures=[rsq, rmse, mae, cor_coef]
        )

        res_dict = Dict()
        res_dict["rsq_mean"] = mean(e.per_fold[1])
        res_dict["rsq_std"] = std(e.per_fold[1])
        res_dict["rmse_mean"] = mean(e.per_fold[2])
        res_dict["rmse_std"] = std(e.per_fold[2])
        res_dict["mae_mean"] = mean(e.per_fold[3])
        res_dict["mae_std"] = std(e.per_fold[3])
        res_dict["r_mean"] = mean(e.per_fold[4])
        res_dict["r_std"] = std(e.per_fold[4])


        @info "\tTraining model on full training set..."
        fit!(mach_occam)

        yhat_train = MLJ.predict(mach_occam, Xtrain)
        yhat_test = MLJ.predict(mach_occam, Xtest)


        # generate plots:
        @info "\tGenerating plots"
        fig = scatter_results(
            ytrain,
            yhat_train,
            ytest,
            yhat_test,
            "$(target_long) ($(units))"
        )

        save(joinpath(path_to_use, "scatterplot-occam__$(suffix).png"), fig)
        save(joinpath(path_to_use, "scatterplot-occam__$(suffix).pdf"), fig)

        fig = quantile_results(
            ytrain,
            yhat_train,
            ytest,
            yhat_test,
            "$(target_long) ($(units))"
        )
        save(joinpath(path_to_use, "qq-occam__$(suffix).png"), fig)
        save(joinpath(path_to_use, "qq-occam__$(suffix).pdf"), fig)

        @info "\tSaving model"
        MLJ.save(joinpath(path_to_use, "$(savename)-occam__$(suffix).jls"), mach_occam)


        # now let's fit a conformal model to estimate an uncertainty bound
        let
            @info "\tEstimating uncertainty with conformal prediction"
            conf_model = conformal_model(mdl; method=:simple_inductive, train_ratio=(8/9), coverage=0.9)
            mach_conf = machine(conf_model, Xtrain, ytrain)
            fit!(mach_conf, verbosity=0)

            yhat_conf = ConformalPrediction.predict(mach_conf, Xtest);
            cov = emp_coverage(yhat_conf, ytest);
            @info "\tEmpirical coverage: $(cov)"
            res_dict["emp_cov"] = cov

            Δy = (yhat_conf[1][2] - yhat_conf[1][1])/2
            res_dict["uncertainty"] = Δy
            @info "\tEstimated uncertainty: $(Δy ) $(units)"
        end


        # save dict of evaluation results...
        open(joinpath(path_to_use, "$(savename)-occam__$(suffix).json"), "w") do f
            JSON.print(f, res_dict)
        end
    end
end




function train_basic(
    X, y,
    idx_train, idx_test,
    longname, savename, mdl,
    target_name, units, target_long,
    outpath;
    suffix="",
    run_occam=false,
    rng=Xoshiro(42),
    )


    @info "\tSetting up save paths"

    outpathdefault = joinpath(outpath, savename, "default")

    path_to_use = outpathdefault
    if !ispath(path_to_use)
        mkpath(path_to_use)
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



    # fit the machine
    @info "\tTraining model"
    mach = machine(mdl, X, y);
    fit!(mach, rows=idx_train)

    # generate predictions
    @info "\tGenerating predictions"
    ytrain = y[idx_train];
    ytest = y[idx_test];
    ŷtrain = MLJ.predict(mach, X[idx_train,:]);
    ŷtest = MLJ.predict(mach, X[idx_test,:]);

    # generate scatterplot
    @info "\tCreating Plots"
    fig = scatter_results(ytrain, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    # generate quantile-quantile plot
    fig = quantile_results(ytrain, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)


    # save the model println("\tSaving the model")
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)


    res_dict = Dict()
    res_dict["rsq_train"] = rsq(ŷtrain, ytrain)
    res_dict["rsq_test"] = rsq(ŷtest, ytest)
    res_dict["rmse_train"] = rmse(ŷtrain, ytrain)
    res_dict["rmse_test"] = rmse(ŷtest, ytest)
    res_dict["mae_train"] = mae(ŷtrain, ytrain)
    res_dict["mae_test"] = mae(ŷtest, ytest)
    res_dict["r_train"] = Statistics.cor(ŷtrain, ytrain)
    res_dict["r_test"] = Statistics.cor(ŷtest, ytest)


    let
        @info "\tEstimating uncertainty with conformal prediction"
        conf_model = conformal_model(mdl; method=:simple_inductive, train_ratio=(8/9), coverage=0.9)
        Xtrain = X[idx_train,:]
        Xtest = X[idx_test,:]

        mach_conf = machine(conf_model, Xtrain, ytrain);
        fit!(mach_conf, verbosity=0)

        yhat_conf = ConformalPrediction.predict(mach_conf, Xtest);
        cov = emp_coverage(yhat_conf, ytest);
        @info "\tEmpirical coverage: $(cov)"
        res_dict["emp_cov"] = cov

        Δy = (yhat_conf[1][2] - yhat_conf[1][1])/2
        res_dict["uncertainty"] = Δy
        @info "\tEstimated uncertainty: $(Δy ) $(units)"
    end

    open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
        JSON.print(f, res_dict)
    end


    # 2. Compute feature importances
    if reports_feature_importances(mdl)
        n_features = 25

        @info "\tComputing feature importances"
        rpt = report(mach);


        fi_pairs = feature_importances(mach) #.model, mach.fitresult, rpt);

        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
        sort!(fi_df, :rel_importance; rev=true)

        CSV.write(joinpath(path_to_use, "importance_ranking__$(suffix).csv"), fi_df)

        rel_importance = fi_df.rel_importance[1:n_features]
        var_names = [name_replacements[s] for s ∈ String.(fi_df.feature_name[1:n_features])]


        var_colors = cgrad([mints_colors[2], mints_colors[1]], n_features)[1:n_features]


        fig = Figure(; resolution=(1000, 1000))
        ax = Axis(
            fig[1, 1],
            yticks=(1:n_features, var_names[end:-1:1]),
            xlabel="Normalized Feature Importance",
            title="$(target_long)",
            yminorgridvisible=false,
            xscale=log10,
        )

        b = barplot!(ax,
            1:n_features,
            rel_importance[end:-1:1] .+ eps(1.0),
            direction=:x,
            color=var_colors
        )

        #xlims!(ax,-0.01, 1.025)
        ylims!(ax, 0.5, n_features + 0.5)

        save(joinpath(path_to_use, "importance_ranking__$(suffix).png"), fig)
        save(joinpath(path_to_use, "importance_ranking__$(suffix).pdf"), fig)

        if run_occam
            #N_features_to_use = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]
            N_features_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,50,100,150]
            r2_vals = Float64[]
            rmse_vals = Float64[]

            for N_features in N_features_to_use
                @info "Evaluating models for $(N_features) features..."
                fi_n = @view fi_df[1:N_features, :]

                pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_n.feature_name)),
                    model=mdl,
                )

                mach_fi = machine(pipe, X, y)
                fit!(mach_fi, rows=idx_train, verbosity=0)

                ŷtrain = MLJ.predict(mach_fi, X[idx_train,:]);
                ŷtest = MLJ.predict(mach_fi, X[idx_test,:]);

                push!(r2_vals, rsq(ŷtest, ytest))
                push!(rmse_vals, rmse(ŷtest, ytest))
            end

            # now let's plot the error as a function of number of features and pick the best
            fig = Figure();
            ax = Axis(fig[1,1], xlabel="N features", ylabel="test R² for $(target_long)")
            ax2 = Axis(fig[1,1], ylabel="test RMSE for $(target_long)", yaxisposition = :right)
            hidespines!(ax2)
            hidexdecorations!(ax2)

            lines!(ax, N_features_to_use, r2_vals, linewidth=3, color=mints_colors[1])
            lines!(ax2, N_features_to_use, rmse_vals, linewidth=3, color=mints_colors[2])

            save(joinpath(path_to_use, "feature-dependence__$(suffix).png"), fig)
            save(joinpath(path_to_use, "feature-dependence__$(suffix).pdf"), fig)


            # train feature reduced model using top 100 features
            # N_final = N_features_to_use[argmax(r2_vals)]
            res_dict = Dict()
            res_dict["n_best"] = N_features_to_use[argmax(r2_vals)]

            N_final = 10

            fi_n = @view fi_df[1:N_final, :]
            pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_n.feature_name)),
                model=mdl,
            )

            mach_fi = machine(pipe, X, y)
            fit!(mach_fi, rows=idx_train, verbosity=0)

            @info "\tGenerating predictions"
            ŷtrain = MLJ.predict(mach_fi, X[idx_train,:]);
            ŷtest = MLJ.predict(mach_fi, X[idx_test,:]);


            fig = scatter_results(ytrain, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")

            save(joinpath(path_to_use, "scatterplot-fi__$(suffix).png"), fig)
            save(joinpath(path_to_use, "scatterplot-fi__$(suffix).pdf"), fig)


            # generate quantile-quantile plot
            fig = quantile_results(ytrain, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")
            save(joinpath(path_to_use, "qq-fi__$(suffix).png"), fig)
            save(joinpath(path_to_use, "qq-fi__$(suffix).pdf"), fig)


            res_dict["rsq_train"] = rsq(ŷtrain, ytrain)
            res_dict["rsq_test"] = rsq(ŷtest, ytest)
            res_dict["rmse_train"] = rmse(ŷtrain, ytrain)
            res_dict["rmse_test"] = rmse(ŷtest, ytest)
            res_dict["mae_train"] = mae(ŷtrain, ytrain)
            res_dict["mae_test"] = mae(ŷtest, ytest)
            res_dict["r_train"] = Statistics.cor(ŷtrain, ytrain)
            res_dict["r_test"] = Statistics.cor(ŷtest, ytest)

            let
                @info "\tEstimating uncertainty with conformal prediction"
                conf_model = conformal_model(pipe; method=:simple_inductive, train_ratio=(8/9), coverage=0.9)
                Xtrain = X[idx_train,:]
                Xtest = X[idx_test,:]

                mach_conf = machine(conf_model, Xtrain, ytrain);
                fit!(mach_conf, verbosity=0)

                yhat_conf = ConformalPrediction.predict(mach_conf, Xtest);
                cov = emp_coverage(yhat_conf, ytest);
                @info "\tEmpirical coverage: $(cov)"
                res_dict["emp_cov"] = cov

                Δy = (yhat_conf[1][2] - yhat_conf[1][1])/2
                res_dict["uncertainty"] = Δy
                @info "\tEstimated uncertainty: $(Δy ) $(units)"
            end

            open(joinpath(path_to_use, "$(savename)-fi__$(suffix).json"), "w") do f
                JSON.print(f, res_dict)
            end
        end
    end

    nothing
end





function train_conformal(
    X, y,
    Xtest, ytest,
    longname, savename, mdl,
    target_name, units, target_long,
    outpath;
    suffix="",
    unc_method=:simple_inductive,
    train_ratio=0.85,
    unc_coverage=0.9,
    rng=Xoshiro(42),
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
    fit!(mach, verbosity=0)

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
    res_dict["r_train"] = Statistics.cor(ŷtrain, y)
    res_dict["r_test"] = Statistics.cor(ŷtest, ytest)
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
        ax = Axis(
            fig[1, 1],
            yticks=(1:n_features, var_names[end:-1:1]),
            xlabel="Normalized Feature Importance",
            title="$(target_long)",
            yminorgridvisible=false,
            xscale=log10,
        )

        b = barplot!(ax,
            1:n_features,
            rel_importance[end:-1:1] .+ eps(1.0),
            direction=:x,
            color=var_colors
        )

        #xlims!(ax,-0.01, 1.025)
        ylims!(ax, 0.5, n_features + 0.5)

        save(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).pdf"), fig)


        # do the same for ONLY reflectances
        ref_vals = Symbol.(["R_" * lpad(i, 3, "0") for i in 1:462])
        fi_pairs_ref = [p for p in fi_pairs if (Symbol(p.first) in ref_vals)]

        fi_df_ref = DataFrame()
        fi_df_ref.feature_name = [x[1] for x ∈ fi_pairs_ref]
        fi_df_ref.rel_importance = [x[2] for x ∈ fi_pairs_ref]
        fi_df_ref.rel_importance .= fi_df_ref.rel_importance ./ maximum(fi_df_ref.rel_importance)
        sort!(fi_df_ref, :rel_importance; rev=true)

        rel_importance = fi_df_ref.rel_importance[1:n_features]
        var_names = [name_replacements[s] for s ∈ String.(fi_df_ref.feature_name[1:n_features])]


        fig = Figure(; resolution=(1000, 1000))
        ax = Axis(
            fig[1, 1],
            yticks=(1:n_features, var_names[end:-1:1]),
            xlabel="Normalized Feature Importance",
            title="$(target_long)",
            yminorgridvisible=false,
            xscale=log10,
        )

        b = barplot!(ax,
                     1:n_features,
                     rel_importance[end:-1:1] .+ eps(1.0),
                     direction=:x,
                     color=var_colors
                     )

        # xlims!(ax, -0.01, 1.025)
        ylims!(ax, 0.5, n_features + 0.5)

        save(joinpath(outpath_featuresreduced, "importance_ranking_refs_only__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "importance_ranking_refs_only__$(suffix).pdf"), fig)


        # now retrain the model with a limited number of features
        N_features_to_use = [1,2, 3, 4, 5,10,15,20,(25:25:(ncol(X)÷25)*25)...]
        r2_vals = Float64[]

        for N_features in N_features_to_use
            @info "Evaluating models for $(N_features) features..."
            fi_n = @view fi_df[1:N_features, :]

            pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_n.feature_name)),
                model=mdl,
            )


            conf_model = conformal_model(pipe; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

            mach_fi = machine(conf_model, X, y)
            fit!(mach_fi, verbosity=0)

            ŷtrain = ConformalPrediction.predict(mach_fi, X);
            ŷtest = ConformalPrediction.predict(mach_fi, Xtest);

            # compute coverage on test set
            cov = emp_coverage(ŷtest, ytest);

            # convert to predictions + uncertainties
            ϵtrain = [abs(f[2] - f[1]) / 2 for f ∈ ŷtrain];
            ŷtrain = mean.(ŷtrain);

            ϵtest = [abs(f[2] - f[1]) / 2 for f ∈ ŷtest]
            ŷtest = mean.(ŷtest)

            res_dict = Dict()
            res_dict["rsq_train"] = rsq(ŷtrain, y)
            res_dict["rsq_test"] = rsq(ŷtest, ytest)
            res_dict["rmse_train"] = rmse(ŷtrain, y)
            res_dict["rmse_test"] = rmse(ŷtest, ytest)
            res_dict["mae_train"] = mae(ŷtrain, y)
            res_dict["mae_test"] = mae(ŷtest, ytest)
            res_dict["r_train"] = Statistics.cor(ŷtrain, y)
            res_dict["r_test"] = Statistics.cor(ŷtest, ytest)
            res_dict["cov"] = cov

            push!(r2_vals, res_dict["rsq_test"])
        end

        # now let's plot the error as a function of number of features and pick the best
        fig = Figure();
        ax = Axis(fig[1,1], xlabel="N features", ylabel="test R² for $(target_long)")
        lines!(ax, N_features_to_use, r2_vals, linewidth=3)
        save(joinpath(outpath_featuresreduced, "feature-dependence__$(suffix).png"), fig)
        save(joinpath(outpath_featuresreduced, "feature-dependence__$(suffix).pdf"), fig)


        # train feature reduced model
        # N_final = N_features_to_use[argmin(rmse_vals)]
        N_final = N_features_to_use[argmax(r2_vals)]

        fi_n = @view fi_df[1:N_final, :]
        pipe = Pipeline(selector=FeatureSelector(features=Symbol.(fi_n.feature_name)),
            model=mdl,
        )


        conf_model = conformal_model(pipe; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

        mach_fi = machine(conf_model, X, y)
        fit!(mach_fi, verbosity=0)


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
        res_dict["r_train"] = Statistics.cor(ŷtrain, y)
        res_dict["r_test"] = Statistics.cor(ŷtest, ytest)
        res_dict["cov"] = cov
        open(joinpath(outpath_featuresreduced, "$(savename)__$(suffix).json"), "w") do f
            JSON.print(f, res_dict)
        end
    end

    nothing
end






function train_hpo(
    X, y,
    idx_train, idx_test,
    longname, savename, packagename,
    target_name, units, target_long,
    mdl,
    outpath;
    nfolds=10,
    nmodels=100,
    rng=Xoshiro(42),
    )

    suffix = "hpo"

    @info "\tSetting up save paths"

    outpathdefault = joinpath(outpath, savename, "default")
    outpath_featuresreduced = joinpath(outpath, savename, "important_only")

    outpath_hpo = joinpath(outpath, savename, "hyperparameter_optimized")
    path_to_use = outpath_hpo

    if !ispath(path_to_use)
        mkpath(path_to_use)
    end


    # 1. Train model
    @info "\tSetting model random seed"


    if :rng in fieldnames(typeof(mdl))
        mdl.rng = rng
    end


    @info "\tVerifying scitype is satisfied"
    # verify scitype is satisfied
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)


    Xtrain = @view X[idx_train, :]
    Xtest = @view X[idx_test, :]

    ytrain = @view y[idx_train]
    ytest = @view y[idx_test]



    # set up hyperparameter ranges
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

    tuning_pipe = TunedModel(
        model=mdl,
        range=rs,
        tuning=tuning,
        measures=[mae, rmse, rsq],
        # resampling=[(idx_train, idx_test),],
        resampling=CV(nfolds=6, rng=rng),
        n=nmodels,
        cache=false,
    )

    mach = machine(tuning_pipe, Xtrain, ytrain; cache=false)  # <- data leakage covered by resampling strategy above

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


    # now we want to train and evaluate a final model
    mdl_final = mdl
    for (key, val) in params_dict
        setproperty!(mdl_final, Symbol(key), val)
    end

    # fit the machine
    @info "\tTraining final hpo model"
    mach = machine(mdl_final, Xtrain, ytrain)
    cv = CV(nfolds=nfolds, rng=rng)

    e = evaluate!(
        mach,
        resampling=cv,
        measures=[rsq, rmse, mae, cor_coef]
    )

    res_dict = Dict()
    res_dict["rsq_mean"] = mean(e.per_fold[1])
    res_dict["rsq_std"] = std(e.per_fold[1])
    res_dict["rmse_mean"] = mean(e.per_fold[2])
    res_dict["rmse_std"] = std(e.per_fold[2])
    res_dict["mae_mean"] = mean(e.per_fold[3])
    res_dict["mae_std"] = std(e.per_fold[3])
    res_dict["r_mean"] = mean(e.per_fold[4])
    res_dict["r_std"] = std(e.per_fold[4])


    @info "\tTraining model on full training set..."
    fit!(mach)

    yhat_train = MLJ.predict(mach, Xtrain)
    yhat_test = MLJ.predict(mach, Xtest)


    # generate plots:
    @info "\tGenerating plots"
    fig = scatter_results(
        ytrain,
        yhat_train,
        ytest,
        yhat_test,
        "$(target_long) ($(units))"
    )

    save(joinpath(path_to_use, "scatterplot__$(suffix).png"), fig)
    save(joinpath(path_to_use, "scatterplot__$(suffix).pdf"), fig)

    fig = quantile_results(
        ytrain,
        yhat_train,
        ytest,
        yhat_test,
        "$(target_long) ($(units))"
    )
    save(joinpath(path_to_use, "qq__$(suffix).png"), fig)
    save(joinpath(path_to_use, "qq__$(suffix).pdf"), fig)

    @info "\tSaving model"
    MLJ.save(joinpath(path_to_use, "$(savename)__$(suffix).jls"), mach)

    # now let's fit a conformal model to estimate an uncertainty bound
    let
        @info "\tEstimating uncertainty with conformal prediction"
        conf_model = conformal_model(mdl_final; method=:simple_inductive, train_ratio=(8/9), coverage=0.9)
        mach_conf = machine(conf_model, Xtrain, ytrain)
        fit!(mach_conf, verbosity=0)

        yhat_conf = ConformalPrediction.predict(mach_conf, Xtest);
        cov = emp_coverage(yhat_conf, ytest);
        @info "\tEmpirical coverage: $(cov)"
        res_dict["emp_cov"] = cov

        Δy = (yhat_conf[1][2] - yhat_conf[1][1])/2
        res_dict["uncertainty"] = Δy
        @info "\tEstimated uncertainty: $(Δy ) $(units)"
    end

    # save dict of evaluation results...
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



    conf_model = conformal_model(stack; method=unc_method, train_ratio=train_ratio, coverage=unc_coverage)

    # fit the machine
    mach = machine(conf_model, X, y; cache=false)


    @info "\tStarting training..."
    fit!(mach, verbosity=0)

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
    res_dict["mae_train"] = mae(ŷtrain, y)
    res_dict["mae_test"] = mae(ŷtest, ytest)
    res_dict["r_train"] = Statistics.cor(ŷtrain, y)
    res_dict["r_test"] = Statistics.cor(ŷtest, ytest)
    res_dict["cov"] = cov
    open(joinpath(path_to_use, "$(savename)__$(suffix).json"), "w") do f
        JSON.print(f, res_dict)
    end
    nothing
end

