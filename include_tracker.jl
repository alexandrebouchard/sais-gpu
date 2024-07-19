

included_files = Set{String}() 

function include(file::String) 
    push!(included_files, file) 
    include(x -> x, file)
end

print_included() = println(join(included_files, ","))
