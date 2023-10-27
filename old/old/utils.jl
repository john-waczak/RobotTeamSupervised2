using HDF5
using Plots
using ProgressMeter
include("config.jl")


default(titlefontsize=20, guidefontsize=16, legendfontsize=16, tickfontsize=16, linewidth=3, margin=3Plots.mm, size=(1200,800))

"""
Generate a plot of Scotty's Ranch from the saved HDF5 file and return the resulting plot object.
"""
function Scotty(pathToScotty::String)
    f = h5open(pathToScotty, "r")
    g = f["Satellite"]
    tile = read(g["img"])
    tile_lat = read(g["lat"])
    tile_lon = read(g["lon"])
    tile = RGB.(tile[1,:,:], tile[2,:,:], tile[3,:,:])
    close(f)

    p = plot(tile_lon,
             tile_lat,
             tile,
             yflip=false,
             xlabel="Longitude",
             ylabel="Latitude",
#             size=(1600, 900),
#             xtickfontsize=18,
#             ytickfontsize=18,
#             xguidefontsize=20,
#             yguidefontsize=20,
#             titlefontsize=22,
             margin=10Plots.mm
             )

    return p
end



"""
Generate a plot of Scotty's Ranch from the saved HDF5 file and return the resulting plot object.
"""
function OakPoint(pathToOakpoint::String)
    f = h5open(pathToOakpoint, "r")
    g = f["Satellite"]
    tile = read(g["img"])
    tile_lat = read(g["lat"])
    tile_lon = read(g["lon"])
    tile = RGB.(tile[1,:,:], tile[2,:,:], tile[3,:,:])
    close(f)

    p = plot(tile_lon,
             tile_lat,
             tile,
             yflip=false,
             xlabel="Longitude",
             ylabel="Latitude",
             #             size=(1600, 900),
             #             xtickfontsize=18,
             #             ytickfontsize=18,
             #             xguidefontsize=20,
             #             yguidefontsize=20,
             #             titlefontsize=22,
             margin=10Plots.mm
             )

    return p
end







"""
Compute the r² value between two vectors.
"""
function r²(y_pred, y_true)
    ȳ = mean(y_true)
    SS_res = sum([(y_true[i]-y_pred[i])^2 for i ∈ 1:length(y_pred)])
    SS_tot = sum([(y_true[i]-ȳ)^2 for i ∈ 1:length(y_pred)])
    return 1 - SS_res/(SS_tot + eps(eltype(y_pred)))
end



"""
Given a fitted machine, print resulting r² values and create scatter and qq plots
"""
function plotTrainingResults(mach, Xtrain, Xtest, target_train, target_test, target, method)
    # compute predictions on training and test set
    pred_train = MLJ.predict(mach, Xtrain)
    pred_test = MLJ.predict(mach, Xtest)

    r2_train = r²(pred_train, target_train)
    r2_test = r²(pred_test, target_test)

    println("r² test: $(r2_test)")
    println("r² train: $(r2_train)")

    # compute min/max for setting plot lims
    min = minimum(vcat(target_train, pred_train, target_test, pred_test))
    max = maximum(vcat(target_train, pred_train, target_test, pred_test))



    p1= scatter(target_train, pred_train,
                framestyle = :box,
                xlabel="$(target) true",
                ylabel="$(target) prediction",
                color=:green,
                alpha=0.5,
                msw=0,
                label="training r²=$(round(r2_train, digits=4))",
                #xtickfontsize=18,
                #ytickfontsize=18,
                #xguidefontsize=20,
                #yguidefontsize=20,
                right_margin = 0Plots.mm,
                top_margin=0Plots.mm,

                #figsize=(16, 12)
                )

    scatter!(target_test, pred_test,
            color=:red,
            alpha=0.5,
            msw=0,
            label="testing r²=$(round(r2_test, digits=4))",
            legend=:topleft)

    plot!([min, max],
          [min, max],
          color=:blue,
          label="1:1")


    nbins=100

    p2 = histogram(target_train,
                   bins=nbins,
                   orientation=:v,
                   framestyle=:none,
                   color=:green,
                   alpha=0.5,
                   label="",
                   bottom_margin=-10Plots.mm
                  )

    histogram!(p2,
               target_test,
               bins= floor(Int, 0.5*nbins),
               orientation=:v,
               color=:red,
               label="",
               bottom_margin=-10Plots.mm
               )



    p3 = histogram(pred_train,
                   bins=nbins,
                   orientation=:h,
                   framestyle=:none,
                   color=:green,
                   alpha=0.5,
                   label="",
                   left_margin=-10Plots.mm
                   )

    histogram!(p3,
               pred_test,
               bins=floor(Int, 0.5*nbins),
               orientation=:h,
               color=:red,
               label="",
               left_margin=-10Plots.mm,
               )

    layout = @layout [top{0.001h}
                      [a            _
                      b{0.85w,0.85h} c]
                    ]
    # titlefontsize=22,
    # , size = (900, 900)

    P1= plot(plot(title="$(method) fit for $(target)", framestyle=:none, bottom_margin=-10Plots.mm), p2, p1, p3, layout = layout, link = :both)

    # now make the quantile-quantile plot
    pqq = qqplot(target_train, pred_train,
                 color=:green,
                 alpha=0.5,
                 label="Training samples",
                 legend=true,
                 #xtickfontsize=18,
                 #ytickfontsize=18,
                 #xguidefontsize=20,
                 #yguidefontsize=20,
                 #titlefontsize=22,
                 #size=(900,900)
                 )
    qqplot!(target_test, pred_test,
            color=:red,
            alpha=0.5,
            label="Testing samples",
            legend=true,
            )
    title!("Quantile-Quantile plot for $(target)")
    xlabel!("$(target) true")
    ylabel!("$(target) prediction")


    return P1, pqq
end




function getTrainingData(path::String, target::Symbol, p::Float64, outpath::String)
    isfile(path)
    targetUnits, targetName = targetsDict[target]

    println("Loading the data")

    # load data into dataframe
    df = DataFrame(CSV.File(path));

    # drop rows with missing data (unable to collocate)
    dropmissing!(df)

    # select only the pre-dye category for training
    gdf = groupby(df, :predye_postdye)

    data = gdf[(predye_postdye="Pre-Dye",)]

    # plot the distribution of target values on the scotty map
    println("Plotting target values on map")
    pmap = Scotty()
    # plot!(pmap, data.longitude, data.latitude, seriestype=:scatter, zcolor=data[!, target], c=:vik, clims=(0, 10), ms=3, markerstrokewidth=0.1, label="", colorbar_title=" \n$(targetName)[$(targetUnits)]")
    plot!(pmap, data.longitude, data.latitude, seriestype=:scatter, zcolor=data[!, target], c=:vik, ms=3, markerstrokewidth=0.1, label="", colorbar_title=" \n$(targetName)[$(targetUnits)]")


    title!("Target distribution for $(targetName)")
    savefig(pmap, joinpath(outpath, "$(string(target))_dataMap.svg"))
    savefig(pmap, joinpath(outpath, "$(string(target))_dataMap.png"))


    println("Separating inputs and targets")
    # separate into Inputs and targets
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


    # split Inputs and Targets
    Targets = DataFrames.select(data, targets_vars)
    X = DataFrames.select(data, Not(vcat(targets_vars, ignorecols)))



    # Convert type from Union{missing, float} to float for sorting
    println("Converting type to Float64")
    @showprogress for col ∈ names(X)
        try
            X[!, col] = convert(Float64, X[:, col]);
        catch e
        end
    end

    # check that scientific types are correct
    schema(X)
    schema(Targets)

    y = Targets[!, target]

    if length(targetsDict[target]) == 3
        ymin = targetsDict[target][3]
        y[y .< ymin] .= 0
    end


    # remove targets with NaN standard deviation
    # remove_vars = []
    # for name ∈ names(X)
    #     if isnan(std(X[!, name]))
    #         push!(remove_vars, name)
    #     end
    # end

    # remove_vars

    # this has rad_MSR_705 in it
    # Xfinal = X[!, Not([:MSR_705, :rad_MSR_705, :heading, :roll, :pitch])]
    Xfinal = X[!, Not([:MSR_705, :rad_MSR_705])]

    # perform a stratified sampling to maintain distribution of values
    println("Performing Stratified train-test split")
    (Xtrain, ytrain), (Xtest, ytest) = stratifiedobs((Xfinal, y), p=p)

    # (Xtrain, ytrain), (Xtest, ytest) = splitobs(shuffleobs((Xfinal, y)), at=p)


    # verify that the sampling maintains the distribution of target values
    println("Plotting train-test histograms")
    histogram(ytrain, bins=100, alpha=0.5, label="train")
    histogram!(ytest, bins=50, label="test")
    title!("$(targetName) [$(targetUnits)]")
    xlabel!("value")
    ylabel!("N")
    savefig(joinpath(outpath, "$(string(target))_train-test_hist.svg"))
    savefig(joinpath(outpath, "$(string(target))_train-test_hist.png"))

    println("Finished!")

    return Xtrain, ytrain, Xtest, ytest
end


