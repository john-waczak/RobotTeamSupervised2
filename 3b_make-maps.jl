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
function in_water(Datacube, varnames; threshold=0.3)
    idx_ndwi = findfirst(varnames .== "NDWI1")
    return findall(Datacube[idx_ndwi,:,:] .> threshold)
end


function get_data_for_pred(h5path, col_names)
    h5 = h5open(h5path, "r")
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    # get indices of relevant varnames
    #idx_varnames = findall([v in col_names for v in varnames])
    idx_varnames = [findfirst(n .== varnames) for n in col_names]

    Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    close(h5)

    # allocate target prediction array
    Y = fill(NaN, size(Latitudes)...)

    # get water pixels
    ij_water = in_water(Data, varnames)

    Data_pred = Data[idx_varnames, ij_water]
    #X_hsi = Tables.table(Data_pred', header=col_names)
    X_hsi = Tables.table(Data_pred', header=varnames[idx_varnames])

    return X_hsi, Y, ij_water, Latitudes, Longitudes
end



RFR = @load RandomForestRegressor pkg=DecisionTree
ETR = @load EvoTreeRegressor pkg=EvoTrees



collection = "Full"
day = "11-23"
flight = "Scotty_2"


day = "12-09"
flight = "NoDye_1"


# flight = "Scotty_1"

# targets_to_map = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490, :TRYP])]

# targets_to_map = [:CDOM, :CO, :SpCond, :HDO, :bgm, :Na, :Cl, :Chl]
targets_to_map = [:bgm]

# targets_to_map = [:CDOM, :CO,]
# targets_to_map = [:CDOM]

# set up relevant paths
hsi_paths = joinpath(hsipath, day)
h5_files = sort_files_numerical!([joinpath(hsi_paths, flight, f) for f in readdir(joinpath(hsi_paths, flight)) if endswith(f, ".h5")])
h5_files_to_use = [f for f in h5_files if !(split(f, "/")[end] in bad_hsi_dict[day][flight])]
h5_date = day


color_clims= Dict(
    "CDOM"  => Dict(
        "11-23" => (20.0, 22.0),
        "12-09" => (17.0, 19.0),
        "12-10" => (16.0, 18.0),
    ),
    "CO" => Dict(
        "11-23" => (25.5, 27.5),
        "12-09" => (23.0, 25.0),
        "12-10" => (24.0, 26.0),
    ),
    "SpCond" => Dict(
        "11-23" => (794, 800),
        "12-09" => (850, 868),
        "12-10" => (855, 870),
    ),
    "HDO" => Dict(
        "11-23" => (8.0, 9.8),
        "12-09" => (13.0,13.9),
        "12-10" => (13.0,13.7),
    ),
    "bgm" => Dict(
        "11-23" => (0.0, 4.0),
        "12-09" => (5.0, 15.0),
        "12-10" => (25,40),
    ),
    "Na" => Dict(
        "11-23" => (200, 400),
        "12-09" => (210, 350),
        "12-10" => (120, 140),
    ),
    "Cl" => Dict(
        "11-23" => (43,58),
        "12-09" => (66,100),
        "12-10" => (50,62),
    ),
    "Chl" => Dict(
        "11-23" => (0.5, 3.5),
        "12-09" => (0.0, 4.0),
        "12-10" => (0.5, 4.0),
    )
)



for target ∈ targets_to_map

    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    ml_models = ["EvoTreeRegressor","EvoTreeRegressor", "EvoTreeRegressor", "RandomForestRegressor", "RandomForestRegressor", "RandomForestRegressor"]
    suffixes = ["vanilla", "vanilla-occam", "hpo", "vanilla", "vanilla-occam", "hpo"]
    model_paths = [
        joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"__vanilla.jls"),
        joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"-occam__vanilla.jls"),
        joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "hyperparameter_optimized", "EvoTreeRegressor"*"__hpo.jls"),
        joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"__vanilla.jls"),
        joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"-occam__vanilla.jls"),
        joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "hyperparameter_optimized", "RandomForestRegressor"*"__hpo.jls"),
    ]


    @info "Loading $(target) data for $(h5_date)"
    data_path = joinpath(datapath, h5_date, target_name, "data")
    X_base = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
    y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
    df_lat_long= CSV.read(joinpath(data_path, "lat_lon.csv"), DataFrame)
    latitudes = df_lat_long.latitudes
    longitudes = df_lat_long.longitudes
    @info "\t...complete"

    for j ∈ 1:length(ml_models)

        suffix = suffixes[j]
        ml_model = ml_models[j]
        model_path = model_paths[j]
        @assert ispath(model_path)

        if occursin("occam", suffix) || occursin("hpo", suffix)
            fi_path = joinpath(datapath, collection, target_name, "models", ml_model, "default", "importance_ranking__vanilla.csv")

            @info "\tLoading feature importances from $(fi_path)..."

            fi_df = CSV.read(fi_path, DataFrame)
            N_final = 25
            fi_n = @view fi_df[1:N_final, :]
            fi_occam = Symbol.(fi_n.feature_name)


            @info"\tRe-slicing data to important columns"
            X = X_base[:, fi_occam]
        else
            X = X_base
        end

        @info "Loading $(target) $(ml_model) mach for $(collection) with $(suffix)"
        mach = machine(model_path)


        # make predictions for collocated data
        yhat_boat = predict(mach, X)


        outpath = joinpath(mapspath, collection, target_name, h5_date, flight)
        if !ispath(outpath)
            mkpath(outpath)
        end

        savepath = joinpath(outpath, ml_model * "__" * suffix)
        ispath(savepath)
        if !ispath(savepath)
            mkpath(savepath)
        end


        fig_tot = cmk.Figure(px_per_unit=5);
        ax_tot = cmk.Axis(fig_tot[1,1], xlabel="longitude", ylabel="latitude", title="Collection Date: $(h5_date)");
        bg_tot = cmk.heatmap!(ax_tot, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

        q_01 = quantile(y, 0.01)
        # q_025 = quantile(y, 0.025)
        q_05 = quantile(y, 0.05)
        # # q_25 = quantile(y, 0.25)
        q_50 = quantile(y, 0.5)
        # # q_75 = quantile(y, 0.75)
        q_95 = quantile(y, 0.95)
        # q_975 = quantile(y, 0.975)
        q_99 = quantile(y, 0.99)

        # clims = color_clims[target_name][h5_date]
        # ncolors = 8 # i.e. a 0.25 spacing
        # cm = cmk.cgrad(:roma, ncolors, rev=true, categorical=true)

        #clims = (q_01, q_99)
        #cm = cmk.cgrad(:roma, [0.0, (q_50-q_05)/(q_95-q_05),1.0], rev=true)
        clims = color_clims[target_name][h5_date]
        cm = cmk.cgrad(:roma, rev=true)

        # cm = cmk.cgrad(:vik, [0.0, (q_50-q_1)/(q_99-q_1), 1.0])
        # cm = cmk.cgrad(:inferno, [0.0, (q_25-q_01)/(q_99-q_01), (q_50-q_01)/(q_99-q_01), (q_75-q_01)/(q_99-q_01), 1.0])
        # cm = cmk.cgrad(:jet)


        nskip=1

        @showprogress for h5path in h5_files_to_use
            map_name = split(split(h5path, "/")[end], ".")[1]

            fig = cmk.Figure();
            ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
            bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

            X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))

            Y_pred = predict(mach, X_hsi)
            Y[ij_water] .= Y_pred



            if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-19", "1-20", "1-21", "1-22", "1-23"]])
                rows_pred = findall([Latitudes[ij] > 33.70152 for ij in ij_water])
                ij_water = ij_water[rows_pred]
                Y_pred = Y_pred[rows_pred]
            end


            # deal with this one HSI
            if h5_date == "11-23" && occursin("2-19", h5path)
                ij_skip = [idx for idx in 1:length(ij_water) if Latitudes[ij_water[idx]] <= 33.7016]
                Y_pred[ij_skip] .= NaN
            end

            sc = cmk.scatter!(ax, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=1)

            cmk.scatter!(ax_tot, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)


            sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=.4, strokecolor=:black);

            cmk.xlims!(ax, -97.7168, -97.7125)
            cmk.ylims!(ax, 33.70075, 33.7035)

            cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])

            fig

            save(joinpath(savepath, map_name * "__" * suffix * ".png"), fig)
        end


        cmk.xlims!(ax_tot, -97.7168, -97.7125)
        cmk.ylims!(ax_tot, 33.70075, 33.7035)
        cb = cmk.Colorbar(fig_tot[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])

        save(joinpath(outpath, ml_model * "__" * suffix * ".png"), fig_tot)

        # add boat data and save again
        sc = cmk.scatter!(ax_tot, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=0.2, strokecolor=:black);

        save(joinpath(outpath, ml_model * "__" * suffix * "-w-boat.png"), fig_tot)
    end
end




