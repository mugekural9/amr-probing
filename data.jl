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
    words
    al_distances
    al_embs
    al
end

function AMR(id, input, nodes, edges, roots)
    entities = []    
    relations = []
    triplets = []
    
    for node in nodes
       push!(entities, node["label"])
    end
    
    for edge in edges
        edgelabel = edge["label"]
        srcid =  edge["source"]
        tgtid =  edge["target"]
        triplet = (srcid + 1, edgelabel, tgtid + 1)
        push!(triplets, triplet)
        push!(relations, edgelabel)
    end

    al_distances = []
    al_embs = []
    al = []
    AMR(id, input, entities, relations, triplets, roots, Any, Any, Any, al_distances, al_embs, al)
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
        embeds = amr.wordembeddings[:,:,layer]
        for align in amr.alignments
            node_id  = align["id"] + 1
            if haskey(align, "label")
                wordembed_id = align["label"] .+ 1
            elseif haskey(align, "values")
                wordembed_id = align["values"][1] .+ 1
            end
            push!(amr.al, (wordembed_id, node_id))
            amr.words = udpipes[amr.id] 
        end
        amr.al_embs = zeros(1024, length(amr.al))
        amr = calculate_distances(amr)
        amr = calc_al_embs(amr)
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


function calculate_distances(amr)
    n = length(amr.entities)
    dicts = []

    for i in 1:n
        push!(dicts, Dict())
        for (s, l, t) in amr.triplets
           if s==i
               dicts[i][t] = 1
           elseif t==i
               dicts[i][s] = 1
           end
       end
       dicts[i][i] = 0
    end

    function calc(src, tgt)
        my_adjs = dicts[src]
        if tgt in keys(my_adjs) return 1; end
        possible_paths = []
        for (k,v) in my_adjs
            for (k2, v2) in dicts[k]
               if k2==tgt
                   tgt_dist = v + v2
                   push!(possible_paths, tgt_dist)
               end
           end
       end
        my_adjs[tgt] = minimum(possible_paths)
    end

    for i in 1:n
        for j in 1:n
            if i != j
                calc(i, j)
            end
        end
    end

    v(x) = x[2]
    all_nodes = collect(1:length(amr.triplets))
    aligned_nodes = v.(amr.al)
    deleted_nodes = [] 
   
    for node in all_nodes
        if !(node in aligned_nodes)
           push!(deleted_nodes, node)
       end
    end

    dd = zeros(length(amr.al), length(amr.al))
    for (j,a) in enumerate(aligned_nodes)
        i=1
       for (k,v) in sort(collect(dicts[a]),by=x->x[1])
           if k in aligned_nodes
               dd[j,i] = v
               i+=1
           end
       end
    end

    amr.al_distances = dd
    return amr
end


function calc_al_embs(amr)
    embs = amr.wordembeddings[:,:,1]
    for (i, (labels,r)) in enumerate(amr.al)
        common = []
        for l in labels
            if isempty(common); common = hcat(embs[:,l])
            else common = hcat(common, embs[:,l]); end
        end
        amr.al_embs[:,i] = sum(common, dims=2)
    end
    return amr
end