# we can use this file to look through the HPO results for each model
# and generate a set of reasonable defaults to use

using Pkg
Pkg.instantiate()

using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using JSON

include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")

# set up the paths
path_to_data = "/scratch/jwaczak/data/analysis_full"
# path_to_data = "/media/john/HSDATA/analysis_full"
ispath(path_to_data)

# # testpath = joinpath(path_to_data, "CDOM", "DecisionTreeRegressor", "hyperparameter_optimized")
# testpath = joinpath(path_to_data, "CDOM", "KNNRegressor", "hyperparameter_optimized")
# readdir(testpath)

# testmach = joinpath(testpath, "KNNRegressor__hpo.jls")
# isfile(testmach)

# model = @load KNNRegressor pkg=NearestNeighborModels

# mach = machine(testmach)




# mdl = model()
# ps = params(fitted_params(mach).best_model)
# for (p, val) ∈ zip(keys(ps), ps)
#     println(p, "\t", val)
#     setproperty!(mdl, Symbol(p), val)
# end


models_to_ignore = ["MLJFlux"]

summary_dict = Dict()
for model_package ∈ keys(hpo_ranges)
    if !(model_package ∈ models_to_ignore)
        models = keys(hpo_ranges[model_package])
        for model_name ∈ models
            df = DataFrame()
            for (target, info) ∈ targetsDict
                try
                    load_string = "model = @load $(model_name) pkg=$(model_package) verbosity=0 add=true"
                    eval(Meta.parse(load_string))

                    path = joinpath(path_to_data, String(target), model_name, "hyperparameter_optimized")
                    fpath = joinpath(path, model_name*"__hpo.jls")

                    mach = machine(fpath)
                    fitted_ps = fitted_params(mach).best_model

                    params_dict = Dict{Any,Any}(:target => target)

                    for hp ∈ hpo_ranges[model_package][model_name]
                        #params_dict[String(hp.hpname)] = getproperty(fitted_ps, hp.hpname)
                        val = getproperty(fitted_ps, hp.hpname)
                        name = hp.hpname
                        params_dict[name] = val
                    end

                    append!(df, DataFrame(params_dict))

                catch e
                    println(e)
                end
            end
            summary_dict[model_name]= df
        end
    end
end


# summary_dict
# summary_dict["XGBoostRegressor"]
# summary_dict["DecisionTreeRegressor"]
# summary_dict["LGBMRegressor"]
# summary_dict["KNNRegressor"]
# summary_dict["RandomForestRegressor"]
# summary_dict["EvoTreeRegressor"]


function get_smart_defaults(df::DataFrame)
    defaults_dict = Dict{Any, Any}()

    for name ∈ names(df)
        if name != :target
            col = df[!, name]
            if eltype(col) == Int64
                defaults_dict[name] = mode(col)
            elseif eltype(col) == Float64
                defaults_dict[name] = mean(col)
            end
        end
    end
    return defaults_dict
end




defaults_dict = Dict("DecisionTree" => Dict("DecisionTreeRegressor" => get_smart_defaults(summary_dict["DecisionTreeRegressor"]),
                                            "RandomForestRegressor" => get_smart_defaults(summary_dict["RandomForestRegressor"]),
                                            ),
                     "XGBoost" => Dict("XGBoostRegressor" => get_smart_defaults(summary_dict["XGBoostRegressor"])),
                     "EvoTrees" => Dict("EvoTreeRegressor" => get_smart_defaults(summary_dict["EvoTreeRegressor"])),
                     "NearestNeighborModels" => Dict("KNNRegressor" => get_smart_defaults(summary_dict["KNNRegressor"])),
                     "LightGBM" => Dict("LGBMRegressor" => get_smart_defaults(summary_dict["LGBMRegressor"])),
                  )


open("smart_hp_defaults.json","w") do f
    JSON.print(f, defaults_dict)
end


