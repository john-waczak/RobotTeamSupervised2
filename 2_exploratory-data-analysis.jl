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

# seed reproducible pseudo-random number generator
rng = Xoshiro(42)


# pull in targets info
include("./config.jl")
include("./viz.jl")


outpath = "/media/teamlary/LabData/RobotTeam/supervised"




targets = String.(keys(targets_dict))

@showprogress for target ∈ targets

    # load in data

    pretty_name = targets_dict[Symbol(target)][2]
    units = targets_dict[Symbol(target)][1]

    data_path = joinpath(outpath, target, "data")
    fig_path = joinpath(outpath, target, "figures", "eda")
    if !ispath(fig_path)
        mkpath(fig_path)
    end

    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    Xtest = CSV.read(joinpath(data_path, "Xtest.csv"), DataFrame)

    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)
    ytest = CSV.read(joinpath(data_path, "ytest.csv"), DataFrame)


    # compute correlation matrix between reflectances

    cm = cor(Matrix(X[:, 1:length(wavelengths)]))

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="λ (nm)", aspect=DataAspect())
    hm = heatmap!(ax, wavelengths, wavelengths, cm, colormap=:inferno)
    cmap = Colorbar(fig[1,2], hm, label="Reflectance Correlation")
    fig

    save(joinpath(fig_path, "feature-correlation.png"), fig)
    save(joinpath(fig_path, "feature-correlation.pdf"), fig)



    # compute correlation between reflectance and target
    cvals = cor(Matrix(X[:, 1:length(wavelengths)]), Matrix(y))[:,1]

    fig = Figure();

    ylabel_title = "Correlation with $(pretty_name)"
    if length(ylabel_title) > 45
        ylabel_title = "Correlation with\n$(pretty_name)"
    end

    ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel=ylabel_title)
    #b = band!(ax, wavelengths, zeros(length(wavelengths)), cvals, color=(mints_colors[1], 0.5))
    hlines!(ax, [0.0], color=(:black, 0.25), linewidth=2)
    l = lines!(ax, wavelengths, cvals, linewidth=3)
    xlims!(ax, wavelengths[1], wavelengths[end])
    fig

    save(joinpath(fig_path, "correlation-width-$(target).png"), fig)
    save(joinpath(fig_path, "correlation-width-$(target).pdf"), fig)

    # generate target histogram/distribution visualization
    ytrain_vals = @view y[:,1]
    ytest_vals = @view ytest[:,1]

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="$(pretty_name) ($(units))", ylabel="Counts")
    h_train = hist!(ax, ytrain_vals; bins=round(Int, sqrt(length(ytrain_vals))), color=(mints_colors[1], 0.75))
    h_test = hist!(ax, ytest_vals; bins=round(Int, sqrt(length(ytest_vals))), color=(mints_colors[2], 0.75))

    axislegend(ax, [h_train, h_test], ["Training", "Testing"]; position=:lt)

    try
        ylims!(ax, 0, nothing)
        # compute 5th and 95th quantiles to set xlims
        xlims!(ax, quantile(ytrain_vals, 0.025), quantile(ytrain_vals, 0.975))
    catch e
        println(e)
    end

    fig

    save(joinpath(fig_path, "$(target)-hist.png"), fig)
    save(joinpath(fig_path, "$(target)-hist.pdf"), fig)
end


