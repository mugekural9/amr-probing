using HDF5, JSON

mutable struct AMR
    id
    input
    entities
    relations
    triplets
    roots
    wordembeddings
    alignments
    distances
    paths
    adjacents
    aligned_embeds
    aligned_nodes
    aligned_words
    al
    words
end

function AMR(id, input, nodes, edges, tops)
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

    aligned_embeds = zeros(1024, length(entities))
    aligned_nodes = []
    aligned_words = []
    aligned_nodes_and_words = []
    AMR(id, input, entities, relations, triplets, tops, Any, Any, Any, Any, Any, aligned_embeds, aligned_nodes, aligned_words, aligned_nodes_and_words, Any)
end


## Text Reader
mutable struct AMRReader
    file::String
    amrset
    ninstances::Int64
    layer
end

function AMRReader(file, layer)
    udpipes = udpipe_reader()
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
                    amr = AMR(amr_instance["id"], amr_instance["input"], amr_instance["nodes"], amr_instance["edges"], amr_instance["tops"])
                    push!(amrset, amr)
                end
            catch
                @warn "Could not parse state. $state"
            end
        end
    end
    amrset = add_alignments(amrset)
    amrset = add_wordembeddings(amrset)
    for (id, amr) in enumerate(amrset)
        amr = calculate_pairwise_distances(amr)
        embeds = amr.wordembeddings[:,:,layer]
        for align in amr.alignments
            node_id  = align["id"] + 1
            if haskey(align, "label")
                wordembed_id = align["label"] .+ 1
            elseif haskey(align, "values")
                wordembed_id = align["values"][1] .+ 1
            end
            push!(amr.aligned_nodes, node_id)
            push!(amr.aligned_words, wordembed_id)
            push!(amr.al, (wordembed_id, node_id))
            #amr.aligned_embeds[:, node_id] =  embeds[:, wordembed_id]
            amr.words = udpipes[amr.id] 
        end
    end
    AMRReader(file, amrset, length(amrset), layer)
end


function AMR_AligmentReader(file)
    alignments = Dict()
    amrset = []
    state = open(file);
    while true
        if eof(state)
            close(state)
            break
        else
            try
                alignment = JSON.parse(state)
                alignments[alignment["id"]] = alignment
            catch
                @warn "Could not parse state. $state"
            end
        end
    end
    return alignments
end

function add_wordembeddings(amrset)
    elmo_embeddings_path= "data/ours/elmo-layers.mrp2019training.amr.wsj.raw.hdf5"
    elmo_layer_no = 1
    embeddings = h5open(elmo_embeddings_path, "r") do file
        read(file)
    end
    for (i,amr) in enumerate(amrset)
       amr.wordembeddings = embeddings[string(i-1)]
    end
    return amrset
end


function add_alignments(amrset)
    alignment_file = "data/mrp/2019/companion/jamr.mrp"
    alignments = AMR_AligmentReader(alignment_file)
    for amr in amrset
        amr.alignments = alignments[amr.id]["nodes"]
    end
    return amrset
end


function generate_adjaceny_matrix(amr)
    adjacents = Dict()
    paths = []
    for id in 1:length(amr.entities)
        adjacents[id] = []
    end
    for trp in amr.triplets
       s, l, t = trp
       sid = findfirst(isequal(s), amr.entities) 
       tid = findfirst(isequal(t), amr.entities) 
       append!(adjacents[sid], tid)
       append!(adjacents[tid], sid)
       push!(paths, (sid,tid))
   end
   paths  = sort(collect(paths),by=x->x[1])
   amr.paths = paths 
   amr.adjacents = adjacents
   return amr
end 


function calculate_pairwise_distances(amr)
    amr = generate_adjaceny_matrix(amr)
    amr.distances = ones(length(amr.entities), length(amr.entities)) * - 1
    for r in 1:size(amr.distances,1)
        for c in 1:size(amr.distances,2)
            dist =  calc_between_nodes(amr,r,c)
            if !isnothing(dist)
                amr.distances[r, c] = dist
                amr.distances[c, r] = dist
            end
        end
    end
    return amr
end 


function calc_between_nodes(amr, p1, p2)
    visited = []
    function helper(dist, src, tgt, visited)
        if src == tgt; return dist; end
        for s in amr.adjacents[src] ## Children
            if s == tgt; return dist+1; end
        end
        for s in amr.adjacents[src]
            if !(s in visited)
                push!(visited, s)
                return helper(dist+1, s, tgt, visited)
            end
        end
    end
    dist = helper(0, p1, p2, visited)
    return dist
end


function udpipe_reader()
  udpipes = Dict()
  file = "data/mrp/2019/companion/udpipe.mrp"
  amrset = []
  state = open(file);
  while true
      if eof(state)
          close(state)
          break
      else
          try
            labels = []
            udpipe_instance = JSON.parse(state)
            for n in udpipe_instance["nodes"]
              push!(labels, n["label"])
            end
            udpipes[udpipe_instance["id"]] = labels #join(labels," ")
          catch
            @warn "Could not parse state. $state"
          end
      end
  end
  return udpipes
end