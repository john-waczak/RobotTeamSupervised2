using CSV, DataFrames
using ProgressMeter
using Statistics
using Random

using CairoMakie
using MintsMakieRecipes

set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)


#using MLJ, Flux, ConformalPrediction
using MLJ

# add in ConformalPrediction via GitHub until my fix ships
using Pkg
Pkg.add(url="https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl.git")
using ConformalPrediction


# for base model prediction:
# using MLJModelInterface: reformat
# predict(conf_model.model, mach.fitresult, reformat(conf_model.model, Xtest)...)



# seed reproducible pseudo-random number generator
rng = Xoshiro(42)

# pull in targets info
include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")

datapath = "/media/teamlary/LabData/RobotTeam/supervised"

summary_file = joinpath(datapath, "vanilla_comparison_full.csv")
isfile(summary_file)

ignore_models = []


# train the vanilla version of all compatible models
if !isfile(summary_file)
    # we will train the vanilla version of all compatible models
    target = :CDOM
    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]


    # load datasets
    data_path = joinpath(datapath, target_name, "data")
    outpath = joinpath(datapath, target_name, "models")


    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]


    filter(model) = model.is_supervised && scitype(y) <: model.target_scitype && scitype(X) <: model.input_scitype && model.prediction_type == :deterministic

    mdls = models(filter)


    mdl_names = [mdl.name for mdl ∈ mdls]
    mdl_packages = [mdl.package_name for mdl ∈ mdls]
    mdl_hr_names = [mdl.human_name for mdl ∈ mdls]

    mlj_interfaces = [load_path(mdl.name, pkg=mdl.package_name) for mdl ∈ mdls]
    mlj_interfaces_base = [split(interf, ".")[1] for interf ∈ mlj_interfaces]

    # run this once to make sure we've got our environment setup
    using Pkg
    Pkg.add(unique(mdl_packages))
    Pkg.add(unique(mlj_interfaces_base))


    # set up the dataframe for storing our results
    summary_df = DataFrame()
    summary_df.model_name = mdl_names
    summary_df.model_package = mdl_packages
    summary_df.model_name_long = mdl_hr_names
    summary_df.model_interface = mlj_interfaces


    # now let's go through each possible target and add blank columns for test and train r² score
    for (target, info) ∈ targets_dict
        println(target)
        summary_df[!, "$(target) train R²"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) test R²"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) train RMSE"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) test RMSE"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) empirical coverage"] = zeros(size(summary_df, 1))
    end

else
    summary_df = CSV.read(summary_file, DataFrame)
end

summary_df


# remove the BetaML models as they take forever for some reason...
summary_df = summary_df[(summary_df.model_package .!= "BetaML") .& (summary_df.model_package .!= "ScikitLearn") .& (summary_df.model_package .!= "PartialLeastSquaresRegressor") .& (summary_df.model_name .!= "LinearRegressor"), :]
#summary_df = summary_df[(summary_df.model_package .!= "ScikitLearn") .& (summary_df.model_package .!= "PartialLeastSquaresRegressor") .& (summary_df.model_name .!= "LinearRegressor"), :]

names(summary_df)

for row ∈ eachrow(summary_df)
    if iszero(collect(row[5:end]))
        load_string = "model = @load $(row.model_name) pkg=$(row.model_package)"
        eval(Meta.parse(load_string))

        # if prediction_type(model) == :probabilistic
        #     pred_function = MLJ.predict_median
        # else
        #     pred_function = MLJ.predict
        # end


        for (target, info) ∈ targets_dict
            if row["$(target) train R²"] == 0.0 && row["$(target) test R²"] == 0.0
                if !(row.model_name ∈ ignore_models)
                    println("Working on model: $(row.model_name)\t target: $(target)")


                    target_name = String(target)
                    target_long = targets_dict[target][2]
                    units = targets_dict[target][1]


                    # load datasets
                    data_path = joinpath(datapath, target_name, "data")
                    outpath = joinpath(datapath, target_name, "models")


                    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
                    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
                    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
                    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]


                    target_name = String(target)
                    data_path = joinpath(datapath, target_name)

                    longname = row.model_name_long
                    savename = row.model_name

                    mdl = model()

                    try
                        r2_train, r2_test, rmse_train, rmse_test, cov = train_basic(
                            X, y,
                            Xtest, ytest,
                            longname, savename, mdl,
                            target_name, units, target_long,
                            outpath;
                            suffix="vanilla__$(row.model_package)",
                            #predict_function = pred_function
                        )

                        # update the DataFrame
                        row["$(target) train R²"] = r2_train
                        row["$(target) test R²"] = r2_test
                        row["$(target) train RMSE"] = rmse_train
                        row["$(target) test RMSE"] = rmse_test
                        row["$(target) empirical coverage"] = cov


                        # incrementally save the dataframe so we don't lose it.
                        CSV.write(joinpath(summary_file), summary_df)

                    catch e
                        println("$(row.model_name) from $(row.model_package) failed...")
                        #println(e)
                    end
                end
            end
        end

    else
        println("$(row.model_name) already explored")
        println(row)
    end
end

