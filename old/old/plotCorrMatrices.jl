using CSV, DataFrames
using Plots, StatsPlots
using Statistics
using Random
Random.seed!(42)

dataPath = "/media/john/HSDATA/georectified/11/23/TargetsAndFeatures.csv"
outpath = "/media/john/HSDATA/georectified/11/23/figures"
isfile(dataPath)
ispath(outpath)

df = DataFrame(CSV.File(dataPath))
size(df)

ignorecols = [:latitude,
              :longitude,
              :ilat,
              :ilon,
              :unix_dt,
              :utc_dt,
              :category,
              :predye_postdye,
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


data = DataFrames.select(df, Not(ignorecols))
Targets = DataFrames.select(df, targets_vars)
X = DataFrames.select(df, Not(vcat(targets_vars, ignorecols)))

targets_cor = cor(Matrix(Targets))
p1= heatmap(targets_cor,
            cb_title = "Pearson Correlation Coefficient",
            xticks=(1:size(Targets)[2], String.(targets_vars)),
            yticks=(1:size(Targets)[2], String.(targets_vars)),
            xrotation=90,
            size=(1200,900),
            margin=5*Plots.mm,
            )
title!("Correlation Matrix for Target Variables")

savefig(p1, joinpath(outpath, "target_cor.png"))
savefig(p1, joinpath(outpath, "target_cor.svg"))


X_cor = cor(Matrix(dropmissing(X)))
p2 = heatmap(names(X),
             names(X),
             X_cor,
             cb_title = "Pearson Correlation Coefficient",
             # xticks=(1:size(X)[2], names(X)),
             # yticks=(1:size(X)[2], names(X)),
             xrotation=90,
             size=(1200,900),
             margin=5*Plots.mm,
            )
title!("Correlation Matrix for input variables")

savefig(p2, joinpath(outpath, "Inputs_cor.png"))
savefig(p2, joinpath(outpath, "Inputs_cor.svg"))


size(data)
names(data)[30]
data_cor = cor(Matrix(dropmissing(data)))
size(data_cor)
res_ref = data_cor[31:31+462, 1:30]
res_rad = data_cor[31+463:31+463+462, 1:30]
res_derived = data_cor[31+463+462:end, 1:30]

size(res)


p3 = heatmap(names(data)[1:30],
             names(data)[31:31+462],
             res_ref,
             cb_title = "Pearson Correlation Coefficient",
             xticks=(0.5:30-0.5, names(data)[1:30]),
             # yticks=(1:462, names(data)[31:31+462]),
             xrotation=90,
             size=(1200, 900),
             margin=5*Plots.mm,
             )

title!("Correlation Matrix: Reflectance vs Targets")

savefig(p3, joinpath(outpath, "reflectance_cor.png"))
savefig(p3, joinpath(outpath, "reflectance_cor.svg"))



p4 = heatmap(names(data)[1:30],
             names(data)[31+463:31+463+462],
             res_rad,
             cb_title = "Pearson Correlation Coefficient",
             xticks=(0.5:30-0.5, names(data)[1:30]),
             xrotation=90,
             size=(1200, 900),
             margin=5*Plots.mm,
             )

title!("Correlation Matrix: Radiance vs Targets")

savefig(p4, joinpath(outpath, "radiance_cor.png"))
savefig(p4, joinpath(outpath, "radiance_cor.svg"))


p5 = heatmap(names(data)[1:30],
             names(data)[31+463+462:end],
             res_derived,
             cb_title = "Pearson Correlation Coefficient",
             xticks=(0.5:30-0.5, names(data)[1:30]),
             yticks=(0.5:length(names(data)[31+463+462:end])-0.5, names(data)[31+463+462:end]),
             xrotation=90,
             size=(1200, 900),
             margin=5*Plots.mm,
             )

title!("Correlation Matrix: Derived vs Targets")

savefig(p5, joinpath(outpath, "derived_cor.png"))
savefig(p5, joinpath(outpath, "derived_cor.svg"))
