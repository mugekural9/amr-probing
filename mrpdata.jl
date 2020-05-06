using HDF5, JSON

mutable struct AMR
    id
    input
    entities
    relations
    triplets
end

function AMR(id, input, nodes, edges)
    entities = []    
    relations = []
    triplets = []
    for node in nodes
       push!(entities, node["label"])
    end

    for edge in edges
       push!(relations, edge["label"])
    end
    
    for edge in edges
        edgelabel = edge["label"]
        srcid= edge["source"];  
        srclabel = nodes[srcid+1]["label"]
        tgtid =  edge["target"];
        tgtlabel = nodes[tgtid+1]["label"]
        triplet = (srclabel, edgelabel, tgtlabel)
        push!(triplets, triplet)
    end
    AMR(id, input, entities, relations, triplets)
end


## Text Reader
mutable struct AMRReader
    file::String
    amrset
    ninstances::Int64
end

function AMRReader(file)
    amrset = []
    state = open(file);
    while true
        if eof(state)
            close(state)
            break
        else
            try
                amr_instance = JSON.parse(state)
                if haskey(amr_instance, "edges")
                    amr = AMR(amr_instance["id"], amr_instance["input"], amr_instance["nodes"], amr_instance["edges"])
                    push!(amrset, amr)
                end
            catch
                @warn "Could not parse state. $state"
            end
        end
    end
    AMRReader(file, amrset, length(amrset))
end