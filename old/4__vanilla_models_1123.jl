using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ProgressMeter
using LaTeXStrings


using StableRNGs
# seed rng for reproducibility
rng = StableRNG(42)


# set the plotting theme
add_mints_theme()
theme(:mints)

# Plots.showtheme(:mints)


include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")

# set default resource for parallelization
MLJ.default_resource(CPUThreads(4))

datapath = "/media/john/HSDATA/datasets/Full"
outpath = "/media/john/HSDATA/analysis_full"

isdir(datapath)
isdir(outpath)

summary_file = joinpath(outpath, "vanilla_comparison_full.csv")
isfile(summary_file)

# ignore_models = ["ConstantRegressor",
#                  "EvoTreeGaussian",
#                  "TheilSenRegressor",
#                  "LinearRegressor"]
ignore_models = []

if !isfile(summary_file)
    # we will train the vanilla version of all compatible models
    target = :CDOM
    target_name = String(target)
    target_long = targetsDict[target][2]
    units = targetsDict[target][1]


    # load datasets
    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    # make sure to include the orientation stuff...
    # use only reflectances as they seem to do fine
    others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]
    refs = [Symbol("λ_$(i)") for i ∈ 1:462]
    X̃ = X[:, vcat(refs, others)]
    X̃test = Xtest[:, vcat(refs, others)]

    filter(model) = model.is_supervised && scitype(y) <: model.target_scitype && scitype(X̃) <: model.input_scitype

    mdls = models(filter)

    mdls[1]

    mdl_names = [mdl.name for mdl ∈ mdls]
    mdl_packages = [mdl.package_name for mdl ∈ mdls]
    mdl_hr_names = [mdl.human_name for mdl ∈ mdls]

    mlj_interfaces = [load_path(mdl.name, pkg=mdl.package_name) for mdl ∈ mdls]
    mlj_interfaces_base = [split(interf, ".")[1] for interf ∈ mlj_interfaces]

    # run this once to make sure we've got our environment setup
    using Pkg
    Pkg.add(unique(mdl_packages))
    Pkg.add(unique(mlj_interfaces_base))


    # now we can free up the space from the example dataset
    X = nothing
    y = nothing
    Xtest = nothing
    ytest = nothing
    GC.gc()


    # set up the dataframe for storing our results
    summary_df = DataFrame()
    summary_df.model_name = mdl_names
    summary_df.model_package = mdl_packages
    summary_df.model_name_long = mdl_hr_names
    summary_df.model_interface = mlj_interfaces


    # now let's go through each possible target and add blank columns for test and train r² score
    for (target, info) ∈ targetsDict
        println(target)
        summary_df[!, "$(target) train r²"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) test r²"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) test RMSE"] = zeros(size(summary_df, 1))
        summary_df[!, "$(target) test MAE"] = zeros(size(summary_df, 1))
    end

else
    summary_df = CSV.File(summary_file) |> DataFrame
end

# remove the BetaML models as they take forever for some reason...
summary_df = summary_df[(summary_df.model_package .!= "BetaML") .& (summary_df.model_package .!= "ScikitLearn") .& (summary_df.model_package .!= "PartialLeastSquaresRegressor") .& (summary_df.model_name .!= "LinearRegressor"), :]


# model = @load ConstantRegressor pkg=MLJModels

# if prediction_type(model) == :probabilistic
#     pred_function = MLJ.predict_median
# else
#     pred_function = MLJ.predict
# end


# target=:CDOM
# target_name = String(target)
# data_path = joinpath(datapath, target_name)

# X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame # y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

# Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
# ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

# # make sure to include the orientation stuff...
# # use only reflectances as they seem to do fine
# others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]
# refs = [Symbol("λ_$(i)") for i ∈ 1:462]
# X̃ = X[:, vcat(refs, others)]
# X̃test = Xtest[:, vcat(refs, others)]

# target_long = targetsDict[target][2]
# units = targetsDict[target][1]
# longname = "Constant Regressor"
# savename = "ConstantRegressor"

# mdl = model()

# r²_train, r²_test, rmse_test, mae_test = train_basic(y, X̃,
#                                                       ytest, X̃test,
#                                                       longname, savename, mdl,
#                                                       target_name, units, target_long,
#                                                       outpath;
#                                                       suffix="vanilla",
#                                                       predict_function = pred_function
#                                                       )




# now let us loop through each row, train the models, and update the dataframe
for row ∈ eachrow(summary_df)

    if iszero(collect(row[5:end]))
        load_string = "model = @load $(row.model_name) pkg=$(row.model_package)"
        eval(Meta.parse(load_string))

        if prediction_type(model) == :probabilistic
            pred_function = MLJ.predict_median
        else
            pred_function = MLJ.predict
        end


        for (target, info) ∈ targetsDict if row["$(target) train r²"] == 0.0 && row["$(target) test r²"] == 0.0
                if !(row.model_name ∈ ignore_models)
                    println("Working on model: $(row.model_name)\t target: $(target)")

                    target_name = String(target)
                    data_path = joinpath(datapath, target_name)

                    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
                    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

                    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
                    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

                    # make sure to include the orientation stuff...
                    # use only reflectances as they seem to do fine
                    others = [:altitude, :pitch, :roll, :heading, :solar_az, :solar_el]
                    refs = [Symbol("λ_$(i)") for i ∈ 1:462]
                    X̃ = X[:, vcat(refs, others)]
                    X̃test = Xtest[:, vcat(refs, others)]

                    target_long = targetsDict[target][2]
                    units = targetsDict[target][1]
                    longname = row.model_name_long
                    savename = row.model_name

                    mdl = model()

                    try
                        r²_train, r²_test, rmse_test, mae_test = train_basic(y, X̃,
                                                                             ytest, X̃test,
                                                                             longname, savename, mdl,
                                                                             target_name, units, target_long,
                                                                             outpath;
                                                                             suffix="vanilla",
                                                                             predict_function = pred_function
                                                                             )

                        # update the DataFrame

                        row["$(target) train r²"] = r²_train
                        row["$(target) test r²"] = r²_test
                        row["$(target) test RMSE"] = rmse_test
                        row["$(target) test MAE"] = mae_test

                        # incrementally save the dataframe so we don't lose it.
                        CSV.write(joinpath(outpath, "vanilla_comparison_full.csv"), summary_df)

                    catch e
                        println(e)
                    end
                end
            end
        end

    else
        println("$(row.model_name) already explored")
        println(row)
    end
end



# now that we've fit the models, let's sort them
cols = names(summary_df)
r²_cols = [col for col ∈ cols if occursin("test r²", col)]
rmse_cols = [col for col ∈ cols if occursin("RMSE", col)]
mae_cols = [col for col ∈ cols if occursin("MAE", col)]

r²_vals = summary_df[:, r²_cols]
rmse_vals = summary_df[:, rmse_cols]
mae_vals = summary_df[:, mae_cols]


using Statistics
summary_df.mean_r² = mean.(eachrow(r²_vals))
summary_df.mean_rmse = mean.(eachrow(rmse_vals))
summary_df.mean_mae = mean.(eachrow(mae_vals))

sort!(summary_df, :mean_r², rev=true)
summary_df[:, [:model_name, :mean_r², :mean_rmse, :mean_mae]]

CSV.write(joinpath(outpath, "vanilla_comparison_full.csv"), summary_df)



# # make heatmap of output
# res = summary_df[!, Not([:model_name, :model_package, :model_name_long, :model_interface])]
# data = Array(res)


# x = [name for name ∈ names(summary_df) if !(name ∈ ["model_name", "model_package", "model_name_long", "model_interface"])]
# x = [name[1:end-3] for name ∈ x]
# y = summary_df.model_name

# # set any zeros to NaN
# data[data .<= 0.0]  .= NaN

# pt = palette([:red, :limegreen], 10)

# size(data)
# size(x)
# size(y)
# heatmap(data,
#         c=pt,
#         clims=(0.0, 1.0),
#         xticks=(1:1:size(x,1)+0.5, x),
#         xrotation=90,
#         yticks=(1:1:size(y,1)+0.5, y),
#         size=(1200,900),
#         yguidefontsize=5,
#         colorbar_title=L"$R^2$ $\in$ $(0,1]",
#         title="Vanilla Models for Full Dataset",
#         grid = :none,
#         minorgrid = :none,
#         tick_direction = :none,
#         minorticks = false,
#         framestyle = :box,
#         )

# savefig(joinpath(outpath, "vanilla_comparison_full.png"))
# savefig(joinpath(outpath, "vanilla_comparison_full.pdf"))
# savefig(joinpath(outpath, "vanilla_comparison_full.svg"))
