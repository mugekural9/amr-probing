using HDF5

mutable struct AMR
    index
    id
    snt
    entities
    relations
    embeddings
end

## Text Reader
mutable struct AMRReader
    file::String
    amrset
    ninstances::Int64
end

function AMRReader(file)
    amrset = []
    entities = []
    relations = []
    i = 0
    state = open(file);
    while true
        if eof(state)
            close(state)
            break
        else
            i +=1 
            line = strip(readline(state))
            idpart = findfirst("::id",line)
            datepart = findfirst("::date",line)
            if !isnothing(idpart)
                id = strip(line[idpart.stop+1: datepart.start-1])
                line2 = strip(readline(state))
                sentpart = findfirst("::snt",line2)
                if !isnothing(sentpart)
                    sent = strip(line2[sentpart.stop+1:end])
                    line = strip(readline(state)) # pass the save-date line
                    while !isempty(line)
                        line = strip(readline(state))
                        if isempty(line) continue; end
                        append!(entities, amrentitydecoder(line))
                        append!(relations, amrrelationdecoder(line))
                    end
                    index = length(amrset)
                    amr = AMR(index, id, sent, entities, relations, readembeddings(index))
                    push!(amrset, amr)
                    entities = []
                    relations= []
                end
            end
        end
    end
    AMRReader(file, amrset, length(amrset))
end


function amrentitydecoder(line)
    vars = []
    rx = r"/ (.+)"
    m = match(rx,line)
    if isnothing(m) return []; end

    varname = m.captures[1]
    
    function helperdecode(varname)
        var1 = varname
        rest = []
        if occursin(":",varname)
           var1 = varname[1:findfirst(" ", varname).start]
           rest = varname[findfirst(" ", varname).stop:end]
        end
        return strip(var1), rest
    end

    while !isnothing(helperdecode(varname))
        var, rest= helperdecode(varname)
        push!(vars, replace(var, ")" => ""))
        rest = varname
        rx = r"/ (.+)"
        m = match(rx,varname)
        if isnothing(m) break; end
        varname = m.captures[1]
    end
    return vars
end



function amrrelationdecoder(line)
    vars = []
    var1part = findfirst(":",line)
    var2part = findfirst("(",line)
    if !isnothing(var1part) && !isnothing(var2part) 
        v1end = var1part[1]
        v2strt = var2part[1]
        push!(vars, strip(line[v1end+1:v2strt-1]))
    end
    return vars
end


function readembeddings(index)
    return embeddings[string(index)]
end


datadir = "data/abstract_meaning_representation_amr_2.0/data/amrs/unsplit"
boltfile = "$datadir/amr-release-2.0-amrs-bolt.txt"
elmofile = "data/elmo/elmo-layers.bolt.hdf5"


global embeddings = h5open(elmofile, "r") do file
    read(file)
end
bolt = AMRReader(boltfile)



