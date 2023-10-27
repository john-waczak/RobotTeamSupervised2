using CSV, DataFrames
using Plots
using HDF5
using ProgressMeter


basepath = "/media/snuc/HSDATA/georectified/11/23/Scotty"
# basepath = "/media/john/HSDATA/georectified/11/23/Scotty"

f = h5open("scotty.h5", "r")
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
         title="CO",
         size=(1600, 900),
         xtickfontsize=18,
         ytickfontsize=18,
         xguidefontsize=20,
         yguidefontsize=20,
         titlefontsize=22,
         margin=10Plots.mm
         )


for (root, dirs, files) ∈ walkdir(basepath)
    for f ∈ files
        #if occursin("1-", root) && !occursin("No", root)
        if occursin("1-", root)
            if occursin("df.csv", f)
                println("Loading ", joinpath(root, f))
                df = DataFrame(CSV.File(joinpath(root, f)))
                println("Filter water out...")
                try
                    X = df[df.mNDWI .> 0.25, :]
                    println("plotting...")
                    plot!(p, X.ilon, X.ilat, seriestype=:scatter, zcolor = X.CO_rfr, c=:vik, clims=(25, 28), alpha=0.7, ms = 1, markerstrokewidth=0, label="" )
                    display(p)
                catch e
                    println(e)
                end
            end
        end
    end
end


title!(p, "Crude Oil 11-23-2020 Flight 1 -- Random Forest")

basepath = "/media/snuc/HSDATA/georectified/11/23"
#basepath = "/media/john/HSDATA/georectified/11/23"
data = DataFrame(CSV.File(joinpath(basepath, "TargetsAndFeatures.csv")))
gdf = groupby(data, :predye_postdye)
gdf[2].predye_postdye[1]
data = gdf[2]
plot!(p, data.longitude, data.latitude, seriestype=:scatter, zcolor=data.CO, c=:vik, clims=(25,28), alpha=0.7, ms=3, markerstrokewidth=0.1, label="Boat data")

maximum(gdf[2].CDOM)
minimum(gdf[2].CDOM)


# display(p)
savefig("./11-23_map_CO_1--rfr.png")


