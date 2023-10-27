using SolarGeometry

"""
    makeDatasets(datapath::String, target::Symbol)

Given a path to the TargetsAndFeatures.csv and a desired target variable. Return dataframes with a stratified split with percentage `p` as the training set.
"""
function makeDatasets(datapath::String, target::Symbol) #, p::Float64)
    # load data
    df = CSV.File(joinpath(datapath, "TargetsAndFeatures.csv")) |> DataFrame
    # get rid of any missing
    dropmissing!(df)

    # group by predye_postdye and only train on pre-dye
    gdf = groupby(df, :predye_postdye)
    data = gdf[(predye_postdye="Pre-Dye",)]


    refs = [Symbol("λ_$(i)") for i ∈ 1:462]  # ignore reflectance values since they haven't been helpful
    # ignored_for_input = [refs..., targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705]
    ignored_for_input = [targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705]

    # split into dev set and holdout
    df, df_test = partition(data,
                            .85;
#                            stratify = data[!,target], # make sure we maintain target distribution
                            rng=42  # set the seed for reproducability
                            )

    # now we further split into targets and features
    y, X = unpack(df, ==(target), col -> !(col ∈ ignored_for_input))
#    yval, Xval= unpack(df_val, ==(target), col -> !(col ∈ ignored_for_input))
    ytest, Xtest = unpack(df_test, ==(target), col -> !(col ∈ ignored_for_input))

    # if there's a third column in the targetsDict, set everything below it to 0.0, i.e. the threshold
    if length(targetsDict[target]) == 3
        ymin = targetsDict[target][3]
        y[y .< ymin] .= 0.0
        ytest[ytest .< ymin] .= 0.0
    end

    return (y, X), (ytest, Xtest) #(yval, Xval), (ytest, Xtest)
end








function makeFullDatasets(datapaths::Array{String}, target::Symbol) #, p::Float64)
    dfs = []
    for datapath ∈ datapaths
        # load data
        df = CSV.File(joinpath(datapath, "TargetsAndFeatures.csv")) |> DataFrame
        # get rid of any missing
        dropmissing!(df)

        # group by predye_postdye and only train on pre-dye
        gdf = groupby(df, :predye_postdye)
        data = gdf[(predye_postdye="Pre-Dye",)]

        push!(dfs, data)
    end


    refs = [Symbol("λ_$(i)") for i ∈ 1:462]  # ignore reflectance values since they haven't been helpful
    # ignored_for_input = [refs..., targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705]
    ignored_for_input = [targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705]


    # create joined df by vertical concatenation
    data = vcat(dfs...)

    # split into dev set and holdout
    df, df_test = partition(data,
                            .9;
#                            stratify = data[!,target], # make sure we maintain target distribution
                            rng=42  # set the seed for reproducability
                            )
    # now we further split into targets and features
    y, X = unpack(df, ==(target), col -> !(col ∈ ignored_for_input))
#    yval, Xval= unpack(df_val, ==(target), col -> !(col ∈ ignored_for_input))
    ytest, Xtest = unpack(df_test, ==(target), col -> !(col ∈ ignored_for_input))

    # if there's a third column in the targetsDict, set everything below it to 0.0
    if length(targetsDict[target]) == 3
        ymin = targetsDict[target][3]
        y[y .< ymin] .= 0.0
        ytest[ytest .< ymin] .= 0.0
    end

    return (y, X), (ytest, Xtest) #(yval, Xval), (ytest, Xtest)

end
