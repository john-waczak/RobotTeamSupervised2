using CSV, DataFrames
using ProgressMeter
using Statistics, StatsBase
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


outpath = "/Users/johnwaczak/data/robot-team/supervised"

collections = ["11-23", "12-09", "12-10", "Full"]
# collections = ["11-23", "Full"]



targets = String.(keys(targets_dict))

for collection in collections
    # @showprogress for target ∈ targets
    @info "$(collection):"
    for target ∈ targets
        @info "\t$(target)"

        # collection = "11-23"
        # target = "CDOM"

        # load in data

        pretty_name = targets_dict[Symbol(target)][2]
        units = targets_dict[Symbol(target)][1]

        data_path = joinpath(outpath, collection, target, "data")
        fig_path = joinpath(outpath, collection, target, "exploratory-data-analysis")
        if !ispath(fig_path)
            mkpath(fig_path)
        end

        @info "\t\tloading data..."
        X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
        y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]

        nbins = 1
        bin_width = maximum(y) - minimum(y)
        try
            nbins, bin_width = get_n_bins(y)
        catch e
            println("Failed to find n_bins")
            nbins = 30
            bin_width = (maximum(y) - minimum(y))/nbins
        end




        # compute correlation matrix between reflectances
        @info "\t\tcomputing inter-feature correlations"
        cm = cor(Matrix(X[:, 1:length(wavelengths)]))

        fig = Figure();
        ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="λ (nm)", aspect=DataAspect())
        hm = heatmap!(ax, wavelengths, wavelengths, cm, colormap=:inferno)
        cmap = Colorbar(fig[1,2], hm, label="Reflectance Correlation")

        save(joinpath(fig_path, "feature-correlation.png"), fig)
        save(joinpath(fig_path, "feature-correlation.pdf"), fig)



        # compute correlation between reflectance and target
        @info "\t\tcomputing wavelength correlations..."
        cvals = cor(Matrix(X[:, 1:length(wavelengths)]), y)[:,1]

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

        save(joinpath(fig_path, "correlation-width-$(target).png"), fig)
        save(joinpath(fig_path, "correlation-width-$(target).pdf"), fig)

        # generate target histogram/distribution visualization
        @info "\t\tgenerating histogram..."

        fig = Figure();

        ax = Axis(fig[1,1], xlabel="$(pretty_name) ($(units))", ylabel="Counts", title="N bins = $(nbins), bin width=$(round(bin_width, digits=3))")
        ax2 = Axis(fig[1,1], yaxisposition=:right)
        linkxaxes!(ax, ax2)

        h = hist!(ax, y; bins=nbins, color=(mints_colors[1], 0.75), normalization=:none)
        d = density!(ax2, y[:,1], color=(mints_colors[2], 0.25), strokecolor=(mints_colors[2],0.75), strokewidth=3)

        # h_train = hist!(ax, ytrain_vals; bins=y_hist.edges, color=(mints_colors[1], 0.75), normalization=:pdf)
        # h_test = hist!(ax, ytest_vals; bins=round(Int, sqrt(length(ytest_vals))), color=(mints_colors[2], 0.75))

        # axislegend(ax, [h_train, h_test], ["Training", "Testing"]; position=:lt)

        # try
        #     ylims!(ax, 0, nothing)
        #     # compute 5th and 95th quantiles to set xlims
        #     xlims!(ax, quantile(ytrain_vals, 0.025), quantile(ytrain_vals, 0.975))
        # catch e
        #     println(e)
        # end


        save(joinpath(fig_path, "$(target)-hist.png"), fig)
        save(joinpath(fig_path, "$(target)-hist.pdf"), fig)


        @info "\t\tfinished!"

        GC.gc()
    end
end
