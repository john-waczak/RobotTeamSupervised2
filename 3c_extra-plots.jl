using HDF5, CSV, DataFrames, JSON, Tables
using MLJ, ConformalPrediction
using Random, Statistics
using ProgressMeter
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


include("./config.jl")
include("./viz.jl")


datapath = "/Users/johnwaczak/data/robot-team/supervised"





df_1123 = CSV.read(joinpath(datapath, "11-23", "CDOM", "data", "X.csv"), DataFrame)
df_1209 = CSV.read(joinpath(datapath, "12-09", "CDOM", "data", "X.csv"), DataFrame)
df_1210 = CSV.read(joinpath(datapath, "12-10", "CDOM", "data", "X.csv"), DataFrame)


names(df_1123)

Σdownwelling_1123 = df_1123.Σdownwelling
Σdownwelling_1209 = df_1209.Σdownwelling
Σdownwelling_1210 = df_1210.Σdownwelling

fig = Figure();
ax = Axis(fig[1,1], xlabel="Total Downwelling Intensity");
d = density!(ax, Σdownwelling_1123)
d2 = density!(ax, Σdownwelling_1209)
d3 = density!(ax, Σdownwelling_1210)


fig

Σdownwelling_1123
Σdownwelling_1209
Σdownwelling_1210

h5_test_path = "/Users/johnwaczak/data/robot-team/processed/hsi/11-23/Scotty_1/Scotty_1-1.h5"
@assert isfile(h5_test_path)

h = h5open(h5_test_path, "r")
Data = h["data-Δx_0.1/Data"][:,:,:];
varnames = h["data-Δx_0.1/varnames"][:];

Data[end-1,:,:]
