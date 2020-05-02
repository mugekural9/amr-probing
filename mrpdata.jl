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



function readembeddings(index)
    return embeddings[string(index)]
end


function get_entities_and_relations(dataset)
    all_entities  = []
    all_relations = []
    all_entities_dict= Dict()  
    all_relations_dict = Dict()  

    for amr in dataset.amrset
       append!(all_entities, amr.entities)
       append!(all_relations, amr.relations)
    end

    for e in all_entities
        if !haskey(all_entities_dict, e)
            all_entities_dict[e] = length(all_entities_dict)+1
        end
    end

    for r in all_relations
        if !haskey(all_relations_dict, r)
            all_relations_dict[r] = length(all_relations_dict)+1
        end
    end

    return all_entities_dict, all_relations_dict
end


amr_file = "data/mrp/2020/cf/sample/amr/wsj.mrp"
amr_dataset = AMRReader(amr_file)

