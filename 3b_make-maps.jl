using HDF5, CSV, DataFrames, JSON, Tables
using MLJ, ConformalPrediction
using Random, Statistics
using ProgressMeter

import CairoMakie as cmk
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




include("./config.jl")
include("./viz.jl")
include("./sat-viz.jl")


datapath = "/media/teamlary/LabData/RobotTeam/supervised"
hsipath = "/media/teamlary/LabData/RobotTeam/processed/hsi"

mapspath = "/media/teamlary/LabData/RobotTeam/maps"
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
scotty_2 = joinpath(hsi_11_23, "Scotty_2")

h5_files = [joinpath(scotty_2, f) for f in readdir(scotty_2) if endswith(f, ".h5")]

# let's add a function to sort them into numerical order
function sort_files_numerical!(flist)
    idx_sort = sortperm([lpad(split(s[1], "-")[end], 2, "0") for s in split.(flist, ".")])
    flist .= flist[idx_sort]
end

sort_files_numerical!(h5_files)



# let's open one and extract the data
# create function to tell if the pixel is on land or not
function in_water(Datacube, varnames; threshold=0.1)
    idx_mndwi = findfirst(varnames .== "mNDWI")
    return findall(Datacube[idx_mndwi,:,:] .> threshold)
end



function get_data_for_pred(h5path, col_names)
    h5 = h5open(h5path, "r")
    varnames = read(h5["data-Δx_0.1/varnames"])
    # get indices of relevant varnames
    idx_varnames = findall([v in col_names for v in varnames])
    Data = read(h5["data-Δx_0.1/Data"])[idx_varnames, :, :]
    IsInbounds = read(h5["data-Δx_0.1/IsInbounds"])
    Longitudes = read(h5["data-Δx_0.1/Longitudes"])
    Latitudes = read(h5["data-Δx_0.1/Latitudes"])
    close(h5)

    # allocate target prediction array
    Y = fill(NaN, size(Latitudes)...)

    # get water pixels
    ij_water = in_water(Data, varnames)

    Data_pred = Data[:, ij_water]
    X_hsi = Tables.table(Data_pred', header=names(X))

    return X_hsi, Y, ij_water, Latitudes, Longitudes
end



# load model
model_collection = "11-23"
ETR = @load EvoTreeRegressor pkg=EvoTrees
model = "EvoTreeRegressor"

targets_to_map = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490])]
targets_to_map = [:CDOM]

target = :CDOM
target_name = String(target)
target_long = targets_dict[target][2]
units = targets_dict[target][1]

# create model path
modelpath = joinpath(datapath, model_collection, target_name, "models", model, "default", model*"__vanilla.jls")
@assert ispath(modelpath)
mach = machine(modelpath)

data_path = joinpath(datapath, collection, target_name, "data")
X = CSV.read(joinpath(data_path, "X.csv"), DataFrame)
y = CSV.read(joinpath(data_path, "y.csv"), DataFrame)[:,1]
df_lat_long= CSV.read(joinpath(data_path, "lat_lon.csv"), DataFrame)
latitudes = df_lat_long.latitudes
longitudes = df_lat_long.longitudes

# make predictions for collocated data
yhat_boat = predict(mach, X)


# plot predictions on dataset
fig = cmk.Figure(;resolution=(1200,500));
ax1 = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude", title="Data");
ax2 = cmk.Axis(fig[1,2], xlabel="longitude", ylabel="latitude", title="Predictions");
bg1 = cmk.heatmap!(ax1, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
bg2 = cmk.heatmap!(ax2, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
sc1 = cmk.scatter!(ax1, longitudes, latitudes, color=y, colorrange=(17.5, 21.5), markersize=5);
sc2 = cmk.scatter!(ax2, longitudes, latitudes, color=yhat_boat, colorrange=(17.5, 21.5), markersize=5);
cb = cmk.Colorbar(fig[1,3], colorrange=(17.5, 21.5), label="CDOM (ppb)")
cmk.xlims!(ax1, -97.7168, -97.7125)
cmk.ylims!(ax1, 33.70075, 33.7035)
cmk.xlims!(ax2, -97.7168, -97.7125)
cmk.ylims!(ax2, 33.70075, 33.7035)
fig

# save("pred_comparison_full.png", fig)

fig = cmk.Figure()
ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude")
h = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img)
cmk.xlims!(ax, -97.7168, -97.7125)
cmk.ylims!(ax, 33.70075, 33.7035)
fig

outpath = joinpath(mapspath, model_collection, target_name, "11-23")
if !ispath(outpath)
    mkpath(outpath)
end


fig = cmk.Figure();
ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
clims = (20.0, 21.5)

@showprogress for h5path in h5_files
#    h5path = h5_files[1]

    fig = cmk.Figure();
    ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
    bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
    clims = (20.0, 21.5)

    X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))
    Y_pred = predict(mach, X_hsi)
    Y[ij_water] .= Y_pred
    #h = cmk.heatmap!(ax, Longitudes, Latitudes, Y, colorrange=clims)

    sc = cmk.scatter!(ax, Longitudes[ij_water], Latitudes[ij_water], color=Y_pred, colormap=:jet, colorrange=clims, markersize=1)

    sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=:jet, colorrange=clims, markersize=4, marker=:rect, markerstrokewidth=1, strokecolor=(:black, 0.5));

    cmk.xlims!(ax, -97.7168, -97.7125)
    cmk.ylims!(ax, 33.70075, 33.7035)

    cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=:jet)

    #cmk.heatmap(Y, colorrange=clims)
    # fig
    #map_name = "map" * lpad(, 3, "0")
    map_name = split(split(h5path, "/")[end], ".")[1]
    save(joinpath(outpath, map_name * ".png"), fig)
end

cmk.xlims!(ax, -97.7168, -97.7125)
cmk.ylims!(ax, 33.70075, 33.7035)
cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=:jet)

fig

save(joinpath(outpath, "flight-1-map.png"), fig)

sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=:jet, colorrange=clims, markersize=4, marker=:rect, markerstrokewidth=1, strokecolor=(:black, 0.5));

fig

save(joinpath(outpath, "flight-1-map_w.png"), fig)




save("cdom-map.png", fig)


# define color limits in advance

xs_1 = 0:0.1:1
ys_1 = 0:0.1:1

xs_2 = 2:0.1:3
ys_2 = 2:0.1:3

zs_1 = [x^2+y for x in xs_1, y in xs_1]
zs_2 = [x^2+y for x in xs_2, y in xs_2]

clims = (min(minimum(zs_1), minimum(zs_2)), max(maximum(zs_1), maximum(zs_2)))

fig = cmk.Figure();
ax = cmk.Axis(fig[1,1], xlabel="x", ylabel="y");
h1 = cmk.heatmap!(ax, xs_1, ys_1, zs_1, colorrange=clims)
h2 = cmk.heatmap!(ax, xs_2, ys_2, zs_2, colorrange=clims)
cb = cmk.Colorbar(fig[1,2], colorrange=clims, label="values")
fig



# for maps output we can have a folder structure like
# /media/teamlary/LabData/RobotTeam/maps/target/model/map.png, etc...



# for base model prediction:
# using MLJModelInterface: reformat
# predict(conf_model.model, mach.fitresult, reformat(conf_model.model, Xtest)...)




# seed reproducible pseudo-random number generator
rng = Xoshiro(42)
