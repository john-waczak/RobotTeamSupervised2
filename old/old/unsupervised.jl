using CSV, DataFrames
using Plots, StatsPlots
using Random
using HDF5
using MLJ
using MLDataUtils
using ProgressMeter
using Statistics
include("mlp.jl")
include("utils.jl")
include("config.jl")
Random.seed!(42)


# load in training data
# basepath = "/media/snuc/HSDATA/georectified/11/23/"
basepath = "/media/john/HSDATA/georectified/11/23/"
outpath = joinpath(basepath, "figures")
ispath(outpath)

ispath(basepath)

# for parallelism
MLJ.default_resource(CPUThreads())



df = DataFrame(CSV.File(joinpath(basepath, "TargetsAndFeatures.csv")))
ignorecols = [:latitude,
              :longitude,
              :ilat,
              :ilon,
              :unix_dt,
              :utc_dt,
              :category,
              :predye_postdye,
              :roll,
              :pitch,
              :heading
              ]

targets_vars = [:Br,
                :CDOM,
                :CO,
                :Ca,
                :Chl,
                :ChlRed,
                :Cl,
                :HDO,
                :HDO_percent,
                :NH4,
                :NO3,
                :Na,
                :OB,
                :RefFuel,
                :SSC,
                :Salinity3488,
                :Salinity3490,
                :SpCond,
                :TDS,
                :TRYP,
                :Temp3488,
                :Temp3489,
                :Temp3490,
                :Turb3488,
                :Turb3489,
                :Turb3490,
                :bg,
                :bgm,
                :pH,
                :pH_mV,
                ]


dropmissing!(df)
Targets = DataFrames.select(df, targets_vars)
X = DataFrames.select(df, Not(vcat(targets_vars, ignorecols)))
Xfinal = X[!, Not([:MSR_705, :rad_MSR_705])]


KMEANS = @load KMeans pkg=Clustering
k = 10
km = KMEANS(k=k)
mach = machine(km, Xfinal) |> fit!

y = MLJ.predict(mach, Xfinal)
#yfinal = convert.(Float64, y)
yfinal = convert.(Int, y)

using Colors
colrs = distinguishable_colors(10)

#cs = [colrs[i] for i ∈ yfinal]

p = Scotty()
for i ∈ 1:k
    idx = (yfinal .== i)
    plot!(p, df.ilon[idx], df.ilat[idx],
          seriestype=:scatter,
          msw=0,
          markerstrokealpha=2,
          ms=1,
          mc=colrs[i],
          label="Cluster $(i)",
          )
end
title!(p, "K-means Clustering of HSI Imagery (k=$(k))")
display(p)

savefig(p, joinpath(outpath, "clustering.png"))
savefig(p, joinpath(outpath, "clustering.svg"))


# plot!(p, df.ilon, df.ilat, seriestype=:scatter, zcolor=yfinal, msw=0, markerstrokealpha=1, ms=1, mc=colrs, label="")

# plot!(p, df.ilon, df.ilat, seriestype=:scatter, msw=0, markerstrokealpha=1, ms=1, mc=cs, label="")
