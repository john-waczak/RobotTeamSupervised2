using MLJ: rsq
using StatsBase: iqr



function scatter_folds(
    y,
    ŷ,
    folds,
    rsq_folds,
    varname,
    )

    rsq_mean = round(mean(rsq_folds), digits=3)
    rsq_std = round(std(rsq_folds), digits=3)
    title="$(length(folds))-Fold Cross Validation, R² = $(rsq_mean) ± $(rsq_std)"

    fig = Figure();
    ga = fig[1, 1] = GridLayout()
    axtop = Axis(
        ga[1, 1];
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        title = title
    )
    axmain = Axis(ga[2, 1], xlabel = "True $(varname)", ylabel = "Predicted $(varname)")
    axright = Axis(ga[2, 2];
                  leftspinevisible = false,
                  rightspinevisible = false,
                  bottomspinevisible = false,
                  topspinevisible = false,
                  )

    linkyaxes!(axmain, axright)
    linkxaxes!(axmain, axtop)

    minval, maxval = extrema([extrema(y)..., extrema(ŷ)...])
    δ_edge = 0.1*(maxval-minval)

    # plot 1:1 line
    l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)

    s_plots = []
    s_labels = []
    for i in 1:length(folds)
        idx_train, idx_test = folds[i]
        s = scatter!(axmain, y[idx_test], ŷ[idx_test], alpha=0.75)
        push!(s_plots, s)
        push!(s_labels, "Fold $(i)")
    end

    leg = axislegend(axmain, [s_plots..., l1], [s_labels..., "1:1"]; position=:lt) #, framevisible=false)


    density!(axtop, y, color=(:gray, 0.5), strokecolor=:gray, strokewidth=2)
    density!(axright, ŷ, direction = :y, color=(:gray, 0.5), strokecolor=:gray, strokewidth=2)

    hidedecorations!(axtop)
    hidedecorations!(axright)
    #leg.tellheight = true
    rowsize!(ga, 1, Relative(0.1))
    colsize!(ga, 2, Relative(0.1))

    colgap!(ga, 0)
    rowgap!(ga, 0)

    xlims!(axmain, minval-δ_edge, maxval+δ_edge)
    ylims!(axmain, minval-δ_edge, maxval+δ_edge)

    fig
    return fig
end


function quantile_folds(
    y,
    ŷ,
    folds,
    rsq_folds,
    varname
    )

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="True $(varname)", ylabel="Predicted $(varname)")

    minval, maxval = extrema([extrema(y)..., extrema(ŷ)...,])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(ax, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)

    q_plots = []
    q_labels = []
    for i in 1:length(folds)
        idx_train, idx_test = folds[i]
        q = qqplot!(ax, y[idx_test], ŷ[idx_test], alpha=0.75)
        push!(q_plots, q)
        push!(q_labels, "Fold $(i)")
    end

    leg = axislegend(ax, [q_plots..., l1], [q_labels..., "1:1"]; position=:lt) #, framevisible=false)

    return fig
end




function scatter_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ga = fig[1, 1] = GridLayout()
    axtop = Axis(ga[1, 1];
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
                )
    axmain = Axis(ga[2, 1], xlabel = "True $(varname)", ylabel = "Predicted $(varname)")
    axright = Axis(ga[2, 2];
                  leftspinevisible = false,
                  rightspinevisible = false,
                  bottomspinevisible = false,
                  topspinevisible = false,
                  )

    linkyaxes!(axmain, axright)
    linkxaxes!(axmain, axtop)

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    s1 = scatter!(axmain, y, ŷ, alpha=0.75)
    s2 = scatter!(axmain, ytest, ŷtest, marker=:rect, alpha=0.75)

    labels=[
        "Training R²=$(round(rsq(ŷ, y), digits=3)) (n=$(length(y)))",
        "Testing   R²=$(round(rsq(ŷtest, ytest), digits=3)) (n=$(length(ytest)))",
        "1:1"
    ]

    # leg = Legend(ga[1, 2], [s1, s2, l1], labels)
    leg = axislegend(axmain, [s1, s2, l1], labels; position=:lt)

    density!(axtop, y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axtop, ytest, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    density!(axright, ŷ, direction = :y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axright, ŷtest, direction = :y, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    hidedecorations!(axtop)
    hidedecorations!(axright)
    #leg.tellheight = true
    rowsize!(ga, 1, Relative(0.1))
    colsize!(ga, 2, Relative(0.1))

    colgap!(ga, 0)
    rowgap!(ga, 0)

    xlims!(axmain, minval-δ_edge, maxval+δ_edge)
    ylims!(axmain, minval-δ_edge, maxval+δ_edge)


    return fig
end



function scatter_results(
    y,
    ŷ,
    ϵ̂,
    ytest,
    ŷtest,
    ϵ̂test,
    varname
    )

    fig = Figure();
    ga = fig[1, 1] = GridLayout()
    axtop = Axis(ga[1, 1];
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
                )
    axmain = Axis(ga[2, 1], xlabel = "True $(varname)", ylabel = "Predicted $(varname)")
    axright = Axis(ga[2, 2];
                  leftspinevisible = false,
                  rightspinevisible = false,
                  bottomspinevisible = false,
                  topspinevisible = false,
                  )

    linkyaxes!(axmain, axright)
    linkxaxes!(axmain, axtop)

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    s1 = scatter!(axmain, y, ŷ, alpha=0.75, color=mints_colors[1])
    e1 = errorbars!(axmain, y, ŷ, ϵ̂, alpha=0.75, color=mints_colors[1])

    s2 = scatter!(axmain, ytest, ŷtest, marker=:rect, alpha=0.75, color=mints_colors[2])
    e2 = errorbars!(axmain, ytest, ŷtest, ϵ̂test, alpha=0.75, color=mints_colors[2])

    labels=[
        "Training R²=$(round(rsq(ŷ, y), digits=3)) (n=$(length(y)))",
        "Testing   R²=$(round(rsq(ŷtest, ytest), digits=3)) (n=$(length(ytest)))",
        "1:1"
    ]

    # leg = Legend(ga[1, 2], [s1, s2, l1], labels)
    leg = axislegend(axmain, [s1, s2, l1], labels; position=:lt)

    density!(axtop, y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axtop, ytest, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    density!(axright, ŷ, direction = :y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axright, ŷtest, direction = :y, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    hidedecorations!(axtop)
    hidedecorations!(axright)
    #leg.tellheight = true
    rowsize!(ga, 1, Relative(0.1))
    colsize!(ga, 2, Relative(0.1))

    colgap!(ga, 0)
    rowgap!(ga, 0)

    xlims!(axmain, minval-δ_edge, maxval+δ_edge)
    ylims!(axmain, minval-δ_edge, maxval+δ_edge)


    return fig
end






function quantile_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="True $(varname)", ylabel="Predicted $(varname)")

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(ax, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    qtrain = qqplot!(ax, y, ŷ, alpha=0.5)
    qtest = qqplot!(ax, ytest, ŷtest, marker=:rect, alpha=0.5)

    leg = axislegend(ax, [qtrain, qtest, l1], ["Training", "Testing", "1:1"]; position=:lt)

    return fig
end





function get_n_bins(x)
    # bin_width = 2*iqr(x)/(length(x)^(1/3))  # Freedman-Diaconis rule
    bin_width = 3.49*std(x)/(length(x)^(1/3)) # Scott's rule
    xmin, xmax = extrema(x)
    nbins = round(Int, (xmax-xmin)/bin_width)

    return nbins, bin_width
end


