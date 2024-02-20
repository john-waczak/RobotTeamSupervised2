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


# see https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude
# for detail
ϕ_scale = 33.70093
m_per_deg = 111412.84*cosd(ϕ_scale) - 93.5*cosd(3*ϕ_scale) + 0.118 * cosd(5*ϕ_scale)
λ_scale_l = -97.7166
λ_scale_r = λ_scale_l + 30/m_per_deg



w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413


satmap = get_background_satmap(w,e,s,n)

# fig = cmk.Figure();
# ax = cmk.Axis(fig[1,1])
# cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# cmk.xlims!(ax, -97.7168, -97.7125)
# cmk.ylims!(ax, 33.70075, 33.7035)

# fig



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
# flight = "Scotty_1"


# day = "12-09"
# flight = "NoDye_1"




# flight = "Scotty_1"

# targets_to_map = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490, :TRYP])]


# targets_to_map = [
#     :Temp3488,
#     :SpCond,
#     :Ca,
#     :HDO,
#     :Cl,
#     :Na,
#     :pH,
#     :bgm,
#     :CDOM,
#     :Chl,
#     :OB,
#     :ChlRed,
#     :CO,
#     :Turb3489
# ]


targets_to_map = [
    :Temp3488,
    :SpCond,
    :Ca,
    :HDO,
    :Cl,
    :Na,
    :pH,
    :bg,
    :bgm,
    :CDOM,
    :Chl,
    :OB,
    :ChlRed,
    :CO,
    :Turb3489,
#    :RefFuel,
]

# targets_to_map = [:HDO, :pH]

# targets_to_map = [:CDOM]


# targets_to_map = [:pH]


# set up relevant paths
hsi_paths = joinpath(hsipath, day)

# h5_files = sort_files_numerical!([joinpath(hsi_paths, flight, f) for f in readdir(joinpath(hsi_paths, flight)) if endswith(f, ".h5")])
# h5_files_to_use = [f for f in h5_files if !(split(f, "/")[end] in bad_hsi_dict[day][flight])]


flight = "Scotty_1"
h5_files_1 = sort_files_numerical!([joinpath(hsi_paths, flight, f) for f in readdir(joinpath(hsi_paths, flight)) if endswith(f, ".h5")])
h5_files_to_use_1 = [f for f in h5_files_1 if !(split(f, "/")[end] in bad_hsi_dict[day][flight])]

flight = "Scotty_2"
h5_files_2 = sort_files_numerical!([joinpath(hsi_paths, flight, f) for f in readdir(joinpath(hsi_paths, flight)) if endswith(f, ".h5")])
h5_files_to_use_2 = [f for f in h5_files_2 if !(split(f, "/")[end] in bad_hsi_dict[day][flight])]


flight = "Scotty_combined"
h5_files_to_use = vcat(h5_files_to_use_1..., h5_files_to_use_2...)


h5_date = day






color_clims= Dict(
    "Temp3488" => Dict(
        "11-23" => (13.25, 13.95),
        "12-09" => (8.84, 9.36),
    ),
    "SpCond" => Dict(
        "11-23" => (783, 802),
        "12-09" => (852, 869),
        # "11-23" => (794, 800),
        # "12-09" => (850, 868),
    ),
    "Ca" => Dict(
        "11-23" => (20, 56),
        "12-09" => (1.0, 3.4),
    ),
    "HDO" => Dict(
        "11-23" => (7.8, 9.6),
        "12-09" => (13.2, 13.85),
        # "11-23" => (8.0, 9.8),
        # "12-09" => (13.0,13.9),
    ),
    "Cl" => Dict(
        "11-23" => (44, 57),
        "12-09" => (68, 96),
        # "11-23" => (43,58),
        # "12-09" => (66,100),
    ),
    "Na" => Dict(
        "11-23" => (200, 380),
        "12-09" => (220, 340),
        # "11-23" => (200, 400),
        # "12-09" => (210, 350),
    ),
    "pH" => Dict(
        "11-23" => (7.95, 8.5),
        #"11-23" => (7.95, 8.65),
        "12-09" => (8.2, 8.6),
    ),
    "bg" => Dict(
        "11-23" => (0, 2.1),
        "12-09" => (0,6),
    ),
    "bgm" => Dict(
        #"11-23" => (1.5, 2.5),
        "11-23" => (0.0, 10.0),
        "12-09" => (25, 40),
        # "11-23" => (0.0, 4.0),
        # "12-09" => (5.0, 15.0),
    ),
    "CDOM" => Dict(
        "11-23" => (20.1, 21.6),
        "12-09" => (16, 19),
        # "11-23" => (20.0, 22.0),
        # "12-09" => (17.0, 19.0),
    ),
    "Chl" => Dict(
        #"11-23" => (1.15, 2.0),
        "11-23" => (1.0, 3.0),
        "12-09" => (0.5, 2.0),
        # "11-23" => (0.5, 3.5),
        # "12-09" => (0.0, 4.0),
    ),
    "OB" => Dict(
        "11-23" => (4.5, 5.0),
        "12-09" => (3.8, 4.4),
    ),
    "ChlRed" => Dict(
        "11-23" => (22, 40),
        "12-09" => (10, 50),
    ),
    "CO" => Dict(
        "11-23" => (25.7, 27.3),
        "12-09" => (23, 25),
        # "11-23" => (25.5, 27.5),
        # "12-09" => (23.0, 25.0),
    ),
    "Turb3489" => Dict(
        "11-23" => (1,25),
        "12-09" => (1,3),
    ),
    "RefFuel" => Dict(
        "11-23" => (1.55, 2.4),
        "12-09" => (1.55, 2.4),
    ),
)



for target ∈ targets_to_map

    target_name = String(target)
    target_long = targets_dict[target][2]
    units = targets_dict[target][1]

    # ml_models = ["EvoTreeRegressor","EvoTreeRegressor", "EvoTreeRegressor", "RandomForestRegressor", "RandomForestRegressor", "RandomForestRegressor"]
    # suffixes = ["vanilla", "vanilla-occam", "hpo", "vanilla", "vanilla-occam", "hpo"]
    # model_paths = [
    #     joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"__vanilla.jls"),
    #     joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "default", "EvoTreeRegressor"*"-occam__vanilla.jls"),
    #     joinpath(datapath, collection, target_name, "models", "EvoTreeRegressor", "hyperparameter_optimized", "EvoTreeRegressor"*"__hpo.jls"),
    #     joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"__vanilla.jls"),
    #     joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"-occam__vanilla.jls"),
    #     joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "hyperparameter_optimized", "RandomForestRegressor"*"__hpo.jls"),
    # ]


    ml_models = ["RandomForestRegressor",]
    suffixes = ["vanilla"]
    model_paths = [
        joinpath(datapath, collection, target_name, "models", "RandomForestRegressor", "default", "RandomForestRegressor"*"__vanilla.jls"),
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
        ax_tot = cmk.Axis(fig_tot[1,1], xlabel="Longitude", ylabel="Latitude", title="Collection Date: $(h5_date)");
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
            ax = cmk.Axis(fig[1,1], xlabel="Longitude", ylabel="Latitude");
            bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

            X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))

            Y_pred = predict(mach, X_hsi)
            Y[ij_water] .= Y_pred



            if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-19", "1-20", "1-21", "1-22", "1-23"]])
                rows_pred = findall([Latitudes[ij] > 33.70152 for ij in ij_water])
                ij_water = ij_water[rows_pred]
                Y_pred = Y_pred[rows_pred]
            end

            if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-3", "2-3"]])
                rows_pred = findall([Longitudes[ij] < -97.7152 for ij in ij_water])
                ij_water = ij_water[rows_pred]
                Y_pred = Y_pred[rows_pred]
            end

            # deal with this one HSI
            if h5_date == "11-23" && occursin("2-19", h5path)
                ij_skip = [idx for idx in 1:length(ij_water) if Latitudes[ij_water[idx]] <= 33.7016]
                Y_pred[ij_skip] .= NaN
            end

            sc = cmk.scatter!(ax, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=1, alpha=0.85)

            cmk.scatter!(ax_tot, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)


            sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=.4, strokecolor=:black);

            cmk.xlims!(ax, -97.7168, -97.7125)
            cmk.ylims!(ax, 33.70075, 33.7035)

            # add 30 meter scale bar
            cmk.lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=:white, linewidth=5)
            cmk.text!(ax, λ_scale_l, ϕ_scale - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)

            cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])

            fig

            save(joinpath(savepath, map_name * "__" * suffix * ".png"), fig)
        end


        cmk.xlims!(ax_tot, -97.7168, -97.7125)
        cmk.ylims!(ax_tot, 33.70075, 33.7035)

        # add 30 meter scale bar
        cmk.lines!(ax_tot, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=:white, linewidth=5)
        cmk.text!(ax_tot, λ_scale_l, ϕ_scale - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)

        cb = cmk.Colorbar(fig_tot[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])

        save(joinpath(outpath, ml_model * "__" * suffix * ".png"), fig_tot)

        # add boat data and save again
        sc = cmk.scatter!(ax_tot, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=0.2, strokecolor=:black);

        save(joinpath(outpath, ml_model * "__" * suffix * "-w-boat.png"), fig_tot)
    end
end




