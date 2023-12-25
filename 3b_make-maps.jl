using HDF5, CSV, DataFrames, JSON, Tables
using MLJ, ConformalPrediction
using Random, Statistics
using ProgressMeter

import CairoMakie as cmk
using MintsMakieRecipes

cmk.set_theme!(mints_theme)
cmk.update_theme!(
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


Δx = 0.1
suffix = "hpo1"

include("./config.jl")
include("./viz.jl")
include("./sat-viz.jl")


datapath = "/Users/johnwaczak/data/robot-team/supervised"
hsipath = "/Users/johnwaczak/data/robot-team/processed/hsi"

mapspath = "/Users/johnwaczak/data/robot-team/maps"
if !ispath(mapspath)
    mkpath(mapspath)
end



w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413


satmap = get_background_satmap(w,e,s,n)


# set up relevant paths
collection = "11-23"
hsi_11_23 = joinpath(hsipath, collection)
flight = "Scotty_2"
scotty_2 = joinpath(hsi_11_23, flight)

h5_files = [joinpath(scotty_2, f) for f in readdir(scotty_2) if endswith(f, ".h5")]

# let's add a function to sort them into numerical order
function sort_files_numerical!(flist)
    idx_sort = sortperm([lpad(split(s[1], "-")[end], 2, "0") for s in split.(flist, ".")])
    flist .= flist[idx_sort]
end

sort_files_numerical!(h5_files)



# let's open one and extract the data
# create function to tell if the pixel is on land or not
function in_water(Datacube, varnames; threshold=0.25)
    idx_mndwi = findfirst(varnames .== "mNDWI")
    return findall(Datacube[idx_mndwi,:,:] .> threshold)
end



function get_data_for_pred(h5path, col_names)
    h5 = h5open(h5path, "r")
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    # get indices of relevant varnames
    idx_varnames = findall([v in col_names for v in varnames])
    Data = read(h5["data-Δx_$(Δx)/Data"])[idx_varnames, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    close(h5)

    # allocate target prediction array
    Y = fill(NaN, size(Latitudes)...)

    # get water pixels
    ij_water = in_water(Data, varnames)

    Data_pred = Data[:, ij_water]
    X_hsi = Tables.table(Data_pred', header=names(X))

    return X_hsi, Y, ij_water, Latitudes, Longitudes
end


RFR = @load RandomForestRegressor pkg=DecisionTree
model = "RandomForestRegressor"
model_collections = ["11-23", "Full"]
suffixes = ["vanilla", "hpo1", "hpo2"]

targets_to_map = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490])]


# model_collections = ["11-23"]
# suffixes = ["vanilla"]
# targets_to_map = [:CDOM]


# h5_files_final = h5_files

h5_files_final = [f for f in h5_files if !any(occursin.(["2-16","2-9"], split(f, "/")[end]))]

for target ∈ targets_to_map
    # target=:CDOM

    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    @info "Loading $(target) data"
    data_path = joinpath(datapath, collection, target_name, "data")
    X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
    df_lat_long= CSV.read(joinpath(data_path, "lat_lon.csv"), DataFrame)
    latitudes = df_lat_long.latitudes
    longitudes = df_lat_long.longitudes


    for collection ∈ model_collections
        for suffix ∈ suffixes

            # collection = "11-23"
            # suffix = "vanilla"

            modelpath = joinpath(datapath, collection, target_name, "models", model, "default", model*"__$(suffix).jls")
            @assert ispath(modelpath)

            @info "Loading $(target) mach for $(collection) with $(suffix)"
            mach = machine(modelpath)

            # make predictions for collocated data
            yhat_boat = predict(mach, X)



            outpath = joinpath(mapspath, collection, target_name, "11-23", flight)
            if !ispath(outpath)
                mkpath(outpath)
            end

            savepath = joinpath(outpath, suffix)
            ispath(savepath)
            if !ispath(savepath)
                mkpath(savepath)
            end


            fig_tot = cmk.Figure(px_per_unit=5);
            ax_tot = cmk.Axis(fig_tot[1,1], xlabel="longitude", ylabel="latitude");
            bg_tot = cmk.heatmap!(ax_tot, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

            q_low = quantile(y, 0.01)
            q_mid = quantile(y, 0.5)
            q_high = quantile(y, 0.99)
            clims = (q_low, q_high)
            cm = cmk.cgrad(:vik, [0.0, (q_mid-q_low)/(q_high-q_low), 1.0])

            nskip=1

            @showprogress for h5path in h5_files_final
                map_name = split(split(h5path, "/")[end], ".")[1]

                fig = cmk.Figure();
                ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
                bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

                X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))

                Y_pred = predict(mach, X_hsi)
                Y[ij_water] .= Y_pred

                sc = cmk.scatter!(ax, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)

                cmk.scatter!(ax_tot, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)


                sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=.4, strokecolor=:black);

                cmk.xlims!(ax, -97.7168, -97.7125)
                cmk.ylims!(ax, 33.70075, 33.7035)

                cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm)

                fig

                save(joinpath(savepath, map_name * "__" * suffix * ".png"), fig)
            end


            cmk.xlims!(ax_tot, -97.7168, -97.7125)
            cmk.ylims!(ax_tot, 33.70075, 33.7035)
            cb = cmk.Colorbar(fig_tot[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm)

            fig_tot

            save(joinpath(outpath, flight * "__" * suffix * ".png"), fig_tot)


            sc = cmk.scatter!(ax_tot, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=0.2, strokecolor=:black);

            save(joinpath(outpath, flight * "__" * suffix * "-w-boat.png"), fig_tot)

            fig_tot
        end
    end
end






