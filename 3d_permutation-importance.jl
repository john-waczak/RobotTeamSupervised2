using CSV, DataFrames
using ProgressMeter
using Statistics
using Random

using Dates
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
using MLJ, Flux
using MLJBase: train_test_pairs

# add in ConformalPrediction via GitHub until my fix ships
# using Pkg
# Pkg.add(url="https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl.git")
using ConformalPrediction
using JSON


include("./config.jl")
include("./viz.jl")
include("./training-functions.jl")



ETR = @load EvoTreeRegressor pkg=EvoTrees
RFR = @load RandomForestRegressor pkg=DecisionTree




# load the CDOM dataset and vanilla RFR & ETR models
target = :CDOM
target_name = String(target)
target_long = targets_dict[target][2]
units = targets_dict[target][1]

collection = "Full"



target = :Turb3489
target_name = String(target)



datapath = "/Users/johnwaczak/data/robot-team/supervised"
data_path = joinpath(datapath, collection, target_name, "data")

etr_path = joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"__vanilla.jls")
rfr_path = joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"__vanilla.jls")

@assert ispath(datapath)
@assert ispath(data_path)
@assert ispath(etr_path)
@assert ispath(rfr_path)


X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
idx_train = CSV.read(joinpath(data_path, "idx_train.csv"), DataFrame)[:,1]
idx_test= CSV.read(joinpath(data_path, "idx_test.csv"), DataFrame)[:,1]


ncol(X)

mean(y)
median(y)
extrema(y)
std(y)

fig,ax,d = density(y[y .< 10])
xlims!(ax, 0, 10)
fig

quantile(y, 0.90)


sum(y .> 100) ./ length(y)

mach_etr = machine(etr_path)
mach_rfr = machine(rfr_path)

Xcal = copy(X[idx_test[1:3000], :])
ycal = copy(y[idx_test[1:3000]])

Xcal2 = copy(Xcal)

# return a vector of feature::Symbol => importance::Real pairs



using Random


# note we are assuming that Xcal is a DataFrame
function permutation_importance(mach, Xcal, ycal; score=rsq, ratio=false)
    features = Symbol.(names(Xcal))
    ycal_pred = predict(mach, Xcal)

    # compute base score
    s_base = score(ycal_pred, ycal)

    fi_pairs = []
    @showprogress for j in 1:ncol(Xcal)
        # copy column so we can reset it after
        col_orig = copy(Xcal[:, j])

        # randomly permute the jth column
        Xcal[:,j] = shuffle(Xcal[:, j])

        # compute new score
        s_perm = rsq(predict(mach, Xcal), ycal)

        # importance is differe
        imp = s_base - s_perm
        if ratio
            # if we want it as a ratio instead...
            imp = s_perm/s_base
        end

        # add to list
        push!(fi_pairs, features[j] => imp)

        # reset random column back to normal
        Xcal[:, j] = col_orig
    end

    return fi_pairs
end



# fi_pairs = permutation_importance(mach_etr, Xcal, ycal)
fi_pairs = permutation_importance(mach_rfr, Xcal, ycal)


fi_df = DataFrame()
fi_df.feature_name = [x[1] for x ∈ fi_pairs]
fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
# fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
sort!(fi_df, :rel_importance; rev=true)


n_features = 25

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
    #xscale=log10,
)

b = barplot!(ax,
             1:n_features,
             rel_importance[end:-1:1] .+ eps(1.0),
             direction=:x,
             color=var_colors
             )

#xlims!(ax,-0.01, 1.025)
ylims!(ax, 0.5, n_features + 0.5)


fig

# save("perm-fi-cdom-etr.pdf", fig)
save("perm-fi-cdom-rfr.pdf", fig)





features

fi_df


scatter(fi_df.rel_importance)


using Statistics
