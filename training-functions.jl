import MLJModelInterface as MMI





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
