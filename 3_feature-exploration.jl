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


using MLJ

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


# pull in targets info
include("./config.jl")
include("./viz.jl")


datapath = "/media/teamlary/LabData/RobotTeam/supervised"


XGBR = @load XGBoostRegressor pkg=XGBoost



# 4. Add Random Forest using sk-learn defaults
mdl = XGBR()
mdl.rng = rng

target = :CDOM

target_name = String(target)
target_long = targets_dict[target][2]
units = targets_dict[target][1]

data_path = joinpath(datapath, target_name, "data")

X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]

Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)
ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)[:,1]

Xcal = CSV.read(joinpath(data_path, "Xcal.csv"), DataFrame)
ycal = CSV.read(joinpath(data_path, "ycal.csv"), DataFrame)[:,1]


scitype(y) <: target_scitype(mdl)
scitype(X) <: input_scitype(mdl)


mach = machine(mdl, X, y)
fit!(mach)


ŷtrain = predict(mach, X)
ŷtest = predict(mach, Xtest)
ŷcal = predict(mach, Xtest)


scatter_results(y, ŷtrain, ytest, ŷtest, "$(target_long) ($(units))")

fi_pairs = feature_importances(mach)
fi_df = DataFrame()
fi_df.feature_name = [x[1] for x ∈ fi_pairs]
fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)

String.(fi_df.feature_name)

n_to_use = 25
rel_importance = fi_df.rel_importance[1:n_to_use]
var_names = [name_replacements[s] for s ∈ String.(fi_df.feature_name[1:n_to_use])]

var_colors = cgrad([mints_colors[2], mints_colors[1]], n_to_use)[1:n_to_use]


fig = Figure(;resolution=(1000,1000));
ax = Axis(fig[1,1],
          yticks=(1:n_to_use, var_names[end:-1:1]),
          xlabel="Normalized Feature Importance",
          title="Feature Importance Ranking for $(target_long)",
          yminorgridvisible=false,
          )

b = barplot!(ax,
             1:n_to_use,
             rel_importance[end:-1:1],
             direction=:x,
             color=var_colors
             )

xlims!(ax, -0.01, 1.025)
ylims!(ax, 0.5, n_to_use+0.5)

fig

