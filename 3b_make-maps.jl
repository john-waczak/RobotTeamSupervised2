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


# let's add a function to sort them into numerical order
function sort_files_numerical!(flist)
    idx_sort = sortperm([lpad(split(s[1], "-")[end], 2, "0") for s in split.(flist, ".")])
    flist .= flist[idx_sort]
end


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
    X_hsi = Tables.table(Data_pred', header=col_names)

    return X_hsi, Y, ij_water, Latitudes, Longitudes
end


RFR = @load RandomForestRegressor pkg=DecisionTree
ETR = @load EvoTreeRegressor pkg=EvoTrees

model_collections = ["Full"]
suffixes = ["vanilla", "vanilla_200", "hpo1", "hpo2", "vanilla"]
ml_models = ["RandomForestRegressor", "RandomForestRegressor", "RandomForestRegressor", "RandomForestRegressor", "EvoTreeRegressor"]


# suffixes = ["hpo1",]
# ml_models = ["RandomForestRegressor",]



# targets_to_map = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490, :TRYP])]
targets_to_map = [:CDOM, :CO, :SpCond, :HDO, :bgm, :Na, :Cl]
# targets_to_map = [:CDOM]

# set up relevant paths
hsi_11_23 = joinpath(hsipath, "11-23")
hsi_12_09 = joinpath(hsipath, "12-09")
hsi_12_10 = joinpath(hsipath, "12-10")

h5_files_11_23 = sort_files_numerical!([joinpath(hsi_11_23, "Scotty_2", f) for f in readdir(joinpath(hsi_11_23, "Scotty_2")) if endswith(f, ".h5")])
h5_files_11_23 = [f for f in h5_files_11_23 if !any(occursin.(["2-9", "2-16", "2-20", "2-21", "2-22", "2-23"], f))]

h5_files_12_09 = sort_files_numerical!([joinpath(hsi_12_09, "NoDye_2", f) for f in readdir(joinpath(hsi_12_09, "NoDye_2")) if endswith(f, ".h5")])

h5_files_12_10 = sort_files_numerical!([joinpath(hsi_12_10, "NoDye_2", f) for f in readdir(joinpath(hsi_12_10, "NoDye_2")) if endswith(f, ".h5")])

h5_files = [h5_files_11_23, h5_files_12_09, h5_files_12_10]
h5_dates = ["11-23", "12-09", "12-10"]

collection = "Full"


for target ∈ targets_to_map

    # target = :CDOM

    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    for i ∈ 1:length(h5_dates)
        # i = 1

        h5_date = h5_dates[i]
        h5_files_to_use = h5_files[i]

        @info "Loading $(target) data for $(h5_date)"
        data_path = joinpath(datapath, h5_date, target_name, "data")
        X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
        y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
        df_lat_long= CSV.read(joinpath(data_path, "lat_lon.csv"), DataFrame)
        latitudes = df_lat_long.latitudes
        longitudes = df_lat_long.longitudes

        for j ∈ 1:length(suffixes)
            # suffixes
            # j = 1

            suffix = suffixes[j]
            model = ml_models[j]

            modelpath = joinpath(datapath, collection, target_name, "models", model, "default", model*"__$(suffix).jls")
            @assert ispath(modelpath)

            @info "Loading $(target) mach for $(collection) with $(suffix)"
            mach = machine(modelpath)

            # make predictions for collocated data
            yhat_boat = predict(mach, X)


            outpath = joinpath(mapspath, collection, target_name, h5_date)
            if !ispath(outpath)
                mkpath(outpath)
            end

            savepath = joinpath(outpath, model * "__" * suffix)
            ispath(savepath)
            if !ispath(savepath)
                mkpath(savepath)
            end


            fig_tot = cmk.Figure(px_per_unit=5);
            ax_tot = cmk.Axis(fig_tot[1,1], xlabel="longitude", ylabel="latitude", title="Collection Date: $(h5_date)");
            bg_tot = cmk.heatmap!(ax_tot, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

            # q_01 = quantile(y, 0.01)
            q_05 = quantile(y, 0.05)
            # q_25 = quantile(y, 0.25)
            # q_50 = quantile(y, 0.5)
            # q_75 = quantile(y, 0.75)
            q_95 = quantile(y, 0.95)
            # q_99 = quantile(y, 0.99)

            clims = (q_05, q_95)

            # cm = cmk.cgrad(:vik, [0.0, (q_50-q_1)/(q_99-q_1), 1.0])
            # cm = cmk.cgrad(:inferno, [0.0, (q_25-q_01)/(q_99-q_01), (q_50-q_01)/(q_99-q_01), (q_75-q_01)/(q_99-q_01), 1.0])
            cm = cmk.cgrad(:jet)


            nskip=1

            @showprogress for h5path in h5_files_to_use
                map_name = split(split(h5path, "/")[end], ".")[1]

                fig = cmk.Figure();
                ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
                bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

                X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))

                Y_pred = predict(mach, X_hsi)
                Y[ij_water] .= Y_pred

                # deal with this one HSI
                if h5_date == "11-23" && occursin("2-19", h5path)
                    ij_skip = [idx for idx in 1:length(ij_water) if Latitudes[ij_water[idx]] <= 33.7016]
                    Y_pred[ij_skip] .= NaN
                end

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

            save(joinpath(outpath, model * "__" * suffix * ".png"), fig_tot)

            # add boat data and save again
            sc = cmk.scatter!(ax_tot, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=0.2, strokecolor=:black);

            save(joinpath(outpath, model * "__" * suffix * "-w-boat.png"), fig_tot)
        end
    end
end






