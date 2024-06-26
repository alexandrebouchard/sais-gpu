all_reports() = [  
        # header with    # lambda expression used to 
        # width of 9     # compute that report item
        "    T     "   => a -> length(a.schedule),
        "    N     "   => a -> n_particles(a.particles),
        "  time(s) "   => a -> a.timing.time, 
        "  allc(B) "   => a -> a.timing.bytes,
        "   ess    "   => a -> ess(a.particles),
        "    Λ     "   => a -> a.barriers.globalbarrier,
    ]


function report(a::AIS, is_first, is_last)
    reports = reports_available(a)
    if is_first 
        Pigeons.header(reports) 
    end
    println(
        join(
            map(
                pair -> Pigeons.render_report_cell(pair[2], a),
                reports),
            " "
        ))
    if is_last
        Pigeons.hr(reports, "─")
    end
    return nothing
end

function reports_available(a::AIS)
    result = Pair[] 
    for pair in all_reports() 
        try 
            (pair[2])(a) 
            push!(result, pair)
        catch 
            # some recorder has not been used, skip
        end
    end
    return result
end
