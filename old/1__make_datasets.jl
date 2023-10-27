using mintsML
using MLJ
using DataFrames, CSV
using ProgressMeter
using SolarGeometry

include("./config.jl")
include("./utils.jl")

outpath = "/media/john/HSDATA/datasets"
ispath(outpath)

function make_datasets(datapath, outpath)
    @showprogress for (target, info) ∈ targetsDict
        # make output
        target_name = String(target)
        date = split(datapath, "/")[end]
        outpath_base = joinpath(outpath, date, target_name)

        # make sure outpath exists
        if !isdir(outpath_base)
            println("making folder $(outpath_base)")
            mkpath(outpath_base)
        end

        (y, X), (ytest, Xtest) = makeDatasets(datapath, target)

        # compute solar azimuth and elevation
        solar_geo = solar_azimuth_altitude.(X.utc_dt, X.ilat, X.ilon, X.altitude)
        az_el = hcat(collect.(solar_geo)...)'
        X.solar_az .= az_el[:, 1]
        X.solar_el .= az_el[:, 2]

        solar_geo = solar_azimuth_altitude.(Xtest.utc_dt, Xtest.ilat, Xtest.ilon, Xtest.altitude)
        az_el = hcat(collect.(solar_geo)...)'
        Xtest.solar_az .= az_el[:, 1]
        Xtest.solar_el .= az_el[:, 2]

        # drop off time, ilat, and ilon from X variables
        X = X[:, Not([:ilat, :ilon, :utc_dt])]
        Xtest = Xtest[:, Not([:ilat, :ilon, :utc_dt])]



        # now we want to save the data.
        CSV.write(joinpath(outpath_base, "X.csv"), X)
        CSV.write(joinpath(outpath_base, "y.csv"), DataFrame(Dict(target => y)))

        # CSV.write(joinpath(outpath_base, "Xval.csv"), Xval)
        # CSV.write(joinpath(outpath_base, "yval.csv"), DataFrame(Dict(target => yval)))

        CSV.write(joinpath(outpath_base, "Xtest.csv"), Xtest)
        CSV.write(joinpath(outpath_base, "ytest.csv"), DataFrame(Dict(target => ytest)))


    end
end

function make_datasets_full(datapaths, outpath)
    @showprogress for (target, info) ∈ targetsDict
        target_name = String(target)
        outpath_base = joinpath(outpath, target_name)
        # make sure outpath exists
        if !isdir(outpath_base)
            mkpath(outpath_base)
        end

        (y, X), (ytest, Xtest) = makeFullDatasets(datapaths, target)

        # compute solar azimuth and elevation
        solar_geo = solar_azimuth_altitude.(X.utc_dt, X.ilat, X.ilon, X.altitude)
        az_el = hcat(collect.(solar_geo)...)'
        X.solar_az = az_el[:, 1]
        X.solar_el = az_el[:, 2]

        solar_geo = solar_azimuth_altitude.(Xtest.utc_dt, Xtest.ilat, Xtest.ilon, Xtest.altitude)
        az_el = hcat(collect.(solar_geo)...)'
        Xtest.solar_az = az_el[:, 1]
        Xtest.solar_el = az_el[:, 2]

        # drop off time, ilat, and ilon from X variables
        X = X[:, Not([:ilat, :ilon, :utc_dt])]
        Xtest = Xtest[:, Not([:ilat, :ilon, :utc_dt])]


        # now we want to save the data.
        CSV.write(joinpath(outpath_base, "X.csv"), X)
        CSV.write(joinpath(outpath_base, "y.csv"), DataFrame(Dict(target => y)))

        # CSV.write(joinpath(outpath_base, "Xval.csv"), Xval)
        # CSV.write(joinpath(outpath_base, "yval.csv"), DataFrame(Dict(target => yval)))

        CSV.write(joinpath(outpath_base, "Xtest.csv"), Xtest)
        CSV.write(joinpath(outpath_base, "ytest.csv"), DataFrame(Dict(target => ytest)))
    end
end



# do it for 11-23
datapath = "/media/john/HSDATA/processed/11-23"
datapath2 = "/media/john/HSDATA/processed/12-09"
datapath3 = "/media/john/HSDATA/processed/12-10"
ispath(datapath)


make_datasets(datapath, outpath)
make_datasets(datapath2, outpath)
make_datasets(datapath3, outpath)


# read in test dataset to make sure it worked
# isfile(joinpath(outpath, "11-23", "CDOM", "X.csv"))
# df = CSV.File(joinpath(outpath, "11-23", "CDOM", "X.csv")) |> DataFrame
# names(df)




# datapaths = [datapath, datapath2, datapath3]
# outpath = joinpath(outpath, "Full")

# make_datasets_full(datapaths, outpath)

