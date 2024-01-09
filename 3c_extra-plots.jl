using CSV, DataFrames, JSON, Tables
using Random, Statistics
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

nb1, _ = get_n_bins(Σdownwelling_1123)
nb2, _ = get_n_bins(Σdownwelling_1209)
nb3, _ = get_n_bins(Σdownwelling_1210)


fig = Figure();
gl = GridLayout(fig[1,1]);

ax_top = Axis(gl[1,1], title="Downwelling Intensity Variation");
ax = Axis(gl[2,1], xlabel="Downwelling Intensity (W/m²)", ylabel="Counts", );

linkxaxes!(ax_top, ax)

hidedecorations!(ax_top)
hidespines!(ax_top)

h1 = hist!(ax, Σdownwelling_1123, bins=nb1, color=(mints_colors[1], 0.7))
h2 = hist!(ax, Σdownwelling_1209, bins=nb2, color=(mints_colors[2], 0.7))
h3 = hist!(ax, Σdownwelling_1210, bins=nb3, color=(mints_colors[3], 0.7))

d1 = density!(ax_top, Σdownwelling_1123, color=(mints_colors[1], 0.25), strokecolor=(mints_colors[1], 0.75), strokewidth=3)
d2 = density!(ax_top, Σdownwelling_1209, color=(mints_colors[2], 0.25), strokecolor=(mints_colors[2], 0.75), strokewidth=3)
d3 = density!(ax_top, Σdownwelling_1210, color=(mints_colors[3], 0.25), strokecolor=(mints_colors[3], 0.75), strokewidth=3)

rowsize!(gl, 1, Relative(0.15))
rowgap!(gl, 1, Relative(0.001))
ylims!(ax, 0, nothing)

leg = axislegend(ax, [h1, h2, h3], ["11-23", "12-09", "12-10"], position=:lt)
fig

save("./paper/figures/results/downwelling-hist.png", fig)
save("./paper/figures/results/downwelling-hist.pdf", fig)



# make another plot of the correlations between all targets to try and explain the
# ability to predict physical parameters like salinity

# load full dataset
function get_targets(target)
    target_name = String.(target)

    df_1123 = CSV.read(joinpath(datapath, "11-23", target_name, "data", "y.csv"), DataFrame)
    df_1209 = CSV.read(joinpath(datapath, "12-09", target_name, "data", "y.csv"), DataFrame)
    df_1210 = CSV.read(joinpath(datapath, "12-10", target_name, "data", "y.csv"), DataFrame)

    return vcat(df_1123, df_1209, df_1210)
end

get_targets(:CDOM)



targets_to_try = [t for t in Symbol.(keys(targets_dict)) if !(t in [:TDS, :Salinity3490, :bg, :Br, :NH4, :Turb3488, :Turb3490])]

targets_to_try = [
    :Temp3488,
    :SpCond,
    :Ca,
    :HDO,
    :Salinity3488,
    :Cl,
    :Na,
    :TRYP,
    :pH,
    :bgm,
    :NO3,
    :CDOM,
    :Chl,
    :OB,
    :ChlRed,
    :CO,
    :Turb3490
]

df_targets = hcat([get_targets(t) for t in targets_to_try]...)

names(df_targets)



cm = cor(Matrix(df_targets[:, :]))

var_names = [targets_dict[t][2] for t in targets_to_try]
n_vars = ncol(df_targets)

var_ticks = (range(start=1, step=1.0, length=n_vars), var_names)

fig = Figure();

ax = Axis(
    fig[1,1],
    xticks=var_ticks,
    xticklabelsize=15,
    yticks=var_ticks,
    yticklabelsize=15,
    xticklabelrotation=π/2
)

cmap = cgrad(:roma, 10, categorical=true, rev=true)
clims=(-1,1)
hm = heatmap!(ax, cm, colormap=cmap, colorrange=clims)
cb = Colorbar(fig[1,2], hm, label="Correlation", ticks=-1:0.2:1)
fig

save("./paper/figures/results/target_correlations.png", fig)
save("./paper/figures/results/target_correlations.pdf", fig)
