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


Δx = 0.5

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


# load model
# model_collection = "11-23"
model_collection = "Full"
ETR = @load EvoTreeRegressor pkg=EvoTrees
# model = "EvoTreeRegressor"
model = "RandomForestRegressor"

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
savepath = joinpath(outpath, flight)
if !ispath(savepath)
    mkpath(savepath)
end






fig_tot = cmk.Figure(px_per_unit=5);
ax_tot = cmk.Axis(fig_tot[1,1], xlabel="longitude", ylabel="latitude");
bg_tot = cmk.heatmap!(ax_tot, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

Δy = 0.25 # ppb
#Δy = 0.15 # ppb

q_low = quantile(y, 0.01)
q_mid = quantile(y, 0.5)
q_high = quantile(y, 0.99)

#clims = (20.0, 22.0)
clims = (q_low, q_high)
#n_colors = length(clims[1]:Δy:clims[2])
#cm = cmk.cgrad(:vik, n_colors-1; categorical=true)
cm = cmk.cgrad(:vik, [0.0, (q_mid-q_low)/(q_high-q_low), 1.0])

nskip=1
h5_files_final = h5_files

#h5_files_final = [f for f in h5_files if !any(occursin.(["2-16", "2-20", "2-21", "2-22", "2-23", "2-28", "2-29"], split(f, "/")[end]))]
#h5_files_final = [f for f in h5_files if !any(occursin.(["2-16", "2-20", "2-21", "2-22", "2-23"], split(f, "/")[end]))]

h5_files_final = [f for f in h5_files if !any(occursin.(["2-16",], split(f, "/")[end]))]

@showprogress for h5path in h5_files_final
    # h5path = h5_files[1]
    map_name = split(split(h5path, "/")[end], ".")[1]

    fig = cmk.Figure();
    ax = cmk.Axis(fig[1,1], xlabel="longitude", ylabel="latitude");
    bg = cmk.heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);

    X_hsi, Y, ij_water, Latitudes, Longitudes = get_data_for_pred(h5path, names(X))

    Y_pred = predict(mach, X_hsi)
    Y[ij_water] .= Y_pred

    # h = cmk.heatmap!(ax, Longitudes, Latitudes, Y, colormap=cm, colorrange=clims)
    # cmk.heatmap!(ax_tot, Longitudes, Latitudes, Y, colormap=cm, colorrange=clims)

    sc = cmk.scatter!(ax, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)
    cmk.scatter!(ax_tot, Longitudes[ij_water[1:nskip:end]], Latitudes[ij_water[1:nskip:end]], color=Y_pred[1:nskip:end], colormap=cm, colorrange=clims, markersize=3)


    sc = cmk.scatter!(ax, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=.4, strokecolor=:black);

    cmk.xlims!(ax, -97.7168, -97.7125)
    cmk.ylims!(ax, 33.70075, 33.7035)

    cb = cmk.Colorbar(fig[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm)

    fig

    save(joinpath(savepath, map_name * ".png"), fig)
end


cmk.xlims!(ax_tot, -97.7168, -97.7125)
cmk.ylims!(ax_tot, 33.70075, 33.7035)
cb = cmk.Colorbar(fig_tot[1,2], label="$(target_long) ($(units))", colorrange=clims, colormap=cm)

fig_tot

save(joinpath(outpath, flight*".png"), fig_tot)


sc = cmk.scatter!(ax_tot, longitudes, latitudes, color=y, colormap=cm, colorrange=clims, markersize=4, marker=:rect, strokewidth=0.2, strokecolor=:black);

save(joinpath(outpath, flight*"-w-boat.png"), fig_tot)

fig_tot






