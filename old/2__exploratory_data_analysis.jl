using mintsML
using Statistics
using Plots, StatsPlots
using CSV, DataFrames
using ProgressMeter


# set the plot theme using our custom theme
add_mints_theme()
theme(:mints)
# visualize our theme:
#Plots.showtheme(:mints)


include("./config.jl")




# Set up paths to data
basepath = "/media/john/HSDATA/processed"
outpath = "/media/john/HSDATA/analysis"

isdir(basepath)
isdir(outpath)

path_1123 = joinpath(basepath, "11-23/TargetsAndFeatures.csv")
path_1209 = joinpath(basepath, "12-09/TargetsAndFeatures.csv")
path_1210 = joinpath(basepath, "12-10/TargetsAndFeatures.csv")
path_0324 = joinpath(basepath, "03-24/TargetsAndFeatures.csv")

isfile(path_1123)


# Load the data into DataFrames
dfs = Dict("11-23" => CSV.File(path_1123) |> DataFrame,
           "12-09" => CSV.File(path_1209) |> DataFrame,
           "12-10" => CSV.File(path_1210) |> DataFrame,
#           "0324" => CSV.File(path_0324) |> DataFrame,  # maybe ignore this one for now
           )

# let's see how many rows each of these has
for (day, df) ∈ dfs
    println("$(day) - nrows: $(nrow(df))")
    println("\t nmissing: $(sum(ismissing.(df[!,:λ_1])))")
end


# first, let's only grab the pre-dye release data. Don't do anything for 0324 since it's weird
for (key, df) ∈ dfs
    if key == "0324"
        continue
    else
        gdf = groupby(df, :predye_postdye);
        dfs[key] = gdf[("Pre-Dye",)];
    end
end


# check again now that we've filtered only the pre-dye data
for (day, df) ∈ dfs
    println("$(day) - nrows: $(nrow(df))")
    println("\t nmissing: $(sum(ismissing.(df[!,:λ_1])))")
end

# rows with missing data correspond to boat points with no matching HSI pixels. We can drop these rows
for (day, df) ∈ dfs
    dropmissing!(df);
end

# check again now that we've dropped the rows
for (day, df) ∈ dfs
    println("$(day) - nrows: $(nrow(df))")
    println("\t nmissing: $(sum(ismissing.(df[!,:λ_1])))")
end

function make_hist_plots(dfs, target, outpath)
    # setup outgoing directory
    if !isdir(joinpath(outpath, String(target)))
        mkdir(joinpath(outpath, String(target)))
    end

    units = targetsDict[target][1]
    longname = targetsDict[target][2]

    xmins = []
    xmaxs = []
    for (key, df) ∈ dfs
        push!(xmins, quantile(df[!, target], 0.05))
        push!(xmaxs, quantile(df[!, target], 0.95))
    end
    histpdf(dfs["11-23"][!, target], xlabel="$(longname) [$(units)]", label="11-23", plot_title="Value Distribution")
    histpdf!(dfs["12-09"][!, target], label="12-09")
    histpdf!(dfs["12-10"][!, target], label="12-10")
    xlims!(minimum(xmins), maximum(xmaxs))

    savefig(joinpath(outpath, String(target), "marginalhist.pdf"))
    savefig(joinpath(outpath, String(target), "marginalhist.png"))
end





# let's figure out the contour plot now
# ---------------------------------------------------------

# 1. sort the dataframes by the target variable
function spectra_by_target(dfs, target, outpath)
    # setup outgoing directory
    if !isdir(joinpath(outpath, String(target)))
        mkdir(joinpath(outpath, String(target)))
    end

    # collect the long form of variable names
    units = targetsDict[target][1]
    longname = targetsDict[target][2]

    for (key, df) ∈ dfs
        sort!(df, :CDOM)
    end

    refs = ["λ_$(i)" for i ∈ 1:462]
    rads = ["λ_$(i)_rad" for i ∈ 1:462]

    # loop through each df and plot the spectra
    for (key, df) ∈ dfs
        # 2. sample reflectance plot
        cvals = df[!, target]
        cmin = minimum(cvals)
        cmax = maximum(cvals)

        cvals = cvals ./ cmax
        cs = cgrad(:vik, cvals)
        C = [cs[val] for val ∈ cvals]

        p1 = plot()
        p2 = plot()
        p3 = scatter([0,0], [0,1], zcolor = [cmin, cmax], xlims=(1,1.1), xticks=:none, yticks=:none, xshowaxis=false, yshowaxis=false, label="", grid=false, colorbar_title="$(longname) [$(units)]")

        for i ∈ 1:100:nrow(df)
            Rads = Array(df[i,rads])
            Refs = Array(df[i,refs])
            plot!(p1, wavelengths, Rads, lw=1, alpha=0.75, color=C[i], label="")
            plot!(p2, wavelengths, Refs, lw=1, alpha=0.75, color=C[i], label="")
        end
        #xlabel!(p1, "λ [nm]")
        xlabel!(p2, "λ [nm]")
        ylabel!(p1, "Radiance")
        ylabel!(p2, "Reflectance")

        layout = @layout[grid(2,1){0.94w} c{0.05w}]

        Pfinal = plot((p1,p2)..., p3, layout=layout, plot_title="Spectra for $(key)")


        savefig(joinpath(outpath, String(target), "spectra_by_target_$(key).pdf"))
        savefig(joinpath(outpath, String(target), "spectra_by_target_$(key).png"))
    end
end



# ---------------------------------------------------------
function make_corr_plots(dfs, outpath)
    # setup outgoing directory
    if !isdir(joinpath(outpath, "Summary"))
        mkdir(joinpath(outpath, "Summary"))
    end


    for (key, df) ∈ dfs
        targets_df = df[!,targets_vars]
        p = corrmatrix(targets_df, size=(1200,1000))
        title!("Target Variable Correlations for $(key)")
        savefig(joinpath(outpath, "Summary", "correlation_matrix_$(key).pdf"))
        savefig(joinpath(outpath, "Summary", "correlation_matrix_$(key).png"))
    end
end



function downwelling_by_target(dfs, target, outpath)
    # setup outgoing directory
    if !isdir(joinpath(outpath, String(target)))
        mkdir(joinpath(outpath, String(target)))
    end

    # collect the long form of variable names
    units = targetsDict[target][1]
    longname = targetsDict[target][2]

    for (key, df) ∈ dfs
        sort!(df, :CDOM)
    end

    wavs = ["λ_downwelling_$(i)" for i ∈ 1:size(downwelling_wavelengths, 1)]

    # loop through each df and plot the spectra
    for (key, df) ∈ dfs
        # 2. sample reflectance plot
        cvals = df[!, target]
        cmin = minimum(cvals)
        cmax = maximum(cvals)

        cvals = cvals ./ cmax
        cs = cgrad(:vik, cvals)
        C = [cs[val] for val ∈ cvals]

        p1 = plot()
        p2 = scatter([0,0], [0,1], zcolor = [cmin, cmax], xlims=(1,1.1), xticks=:none, yticks=:none, xshowaxis=false, yshowaxis=false, label="", grid=false, colorbar_title="$(longname) [$(units)]")

        for i ∈ 1:100:nrow(df)
            downwells = Array(df[i, wavs])
            plot!(p1, downwelling_wavelengths[30:end], downwells[30:end], lw=1, alpha=0.75, color=C[i], label="")
        end
        #xlabel!(p1, "λ [nm]")
        xlabel!(p1, "λ [nm]")
        ylabel!(p1, "Downwelling Irradiance")

        layout = @layout[a{0.94w} b{0.05w}]

        Pfinal = plot(p1,p2, layout=layout, plot_title="Downwelling Spectra for $(key)")

        savefig(joinpath(outpath, String(target), "downwelling_by_target_$(key).pdf"))
        savefig(joinpath(outpath, String(target), "downwelling_by_target_$(key).png"))
    end
end





# ---------------------------------------------------------
# let's loop through each of the target variables and generate some statistics
@showprogress for target ∈ targets_vars
    try
        make_hist_plots(dfs, target, outpath)
    catch e
        println(e)
    end

    try
        spectra_by_target(dfs, target, outpath)
    catch e
        println(e)
    end


    try
        downwelling_by_target(dfs, target, outpath)
    catch e
        println(e)
    end


end
make_corr_plots(dfs, outpath)

