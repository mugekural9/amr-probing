include("probe.jl")
include("train.jl")

using HDF5, JSON

amr_file = "../mrp/2020/cf/training/amr.mrp"
udpipe_file="../mrp/2020/cf/companion/udpipe.mrp"
alignment_file="../mrp/2020/cf/companion/jamr.mrp"
embeddings_file="data/raw.trn.mrp20-amr.elmo-layers.hdf5"

 mutable struct AMR
      id
      input
      nodes
      edges
      triplets
      roots
      wordembeddings
      nodeembeddings
      nodes_to_anchors
      alignments
      snt_tokens
      distance_matrix
  end
 
 function AMR(id, input, nodes, edges, roots)
      _nodes = []
      _edges = []
      triplets = []
 
      for node in nodes
         push!(_nodes, node["label"])
      end
      for edge in edges
          edgelabel = edge["label"]
          srcid =  edge["source"]
          tgtid =  edge["target"]
          triplet = (srcid + 1, edgelabel, tgtid + 1)
          push!(triplets, triplet)
          push!(_edges, edgelabel)
      end

      distance_matrix = zeros(length(nodes), length(nodes))
      AMR(id, input, _nodes, _edges, triplets, roots, Any, Any, Any, Any, Any, distance_matrix)
  end



## Text Reader
mutable struct AMRReader
    file::String
    amrs
    ninstances::Int64
    layer
end

function AMRReader(file)
    amrs = []
    state = open(file);
    i=0
    while true
        if eof(state)
            close(state)
            break
        else
           try
                amr_instance = JSON.parse(state)
                if haskey(amr_instance, "edges")
                    amr = AMR(amr_instance["id"], amr_instance["input"], amr_instance["nodes"], amr_instance["edges"], amr_instance["tops"])
                    push!(amrs, amr)
                end
            catch
                @warn "Could not parse state. $state"
            end
      end
    end
    return amrs
end

function UdpipeReader(file)
    sentence_tokens = Dict()
    state = open(file);
    i=0
    while true
        if eof(state)
            close(state)
            break
        else
           try
               tokens = []
               instance = JSON.parse(state)
               if haskey(instance, "nodes")
                   for node in instance["nodes"]
                       push!(tokens, node["label"])
                    end
               end
               sentence_tokens[instance["id"]] = tokens
            catch
                @warn "Could not parse state. $state"
            end
      end
    end
    return sentence_tokens
end



function get_word_embeddings(embeddings_file)
    embeddings = h5open(embeddings_file, "r") do file
        read(file)
    end
    return embeddings
end


function get_alignments(alignment_file)
    alignments = Dict()
    state = open(alignment_file);
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


function append_alignments_to_amrs(amrs, alignments)
    for amr in amrs
        if haskey(alignments[amr.id], "nodes")
            amr.alignments = deepcopy(alignments[amr.id]["nodes"])
            ## fix index incompatibility
            for align in amr.alignments
                align["id"] = align["id"] + 1
                if haskey(align, "anchors")
                    for alg in align["anchors"]
                        alg["#"] =  alg["#"] +1
                    end
                end
                if haskey(align, "anchorings")
                    for alg in align["anchorings"]
                        for alg2 in alg
                            alg2["#"] =  alg2["#"] +1
                        end
                    end
                end
            end
        end
   end
end

""" Append word embeddings and snt tokens as well."""
function append_wordembeddings_to_amrs(amrs, snt_tokens, embeddings)
    for (i,amr) in enumerate(amrs)
        amr.wordembeddings = embeddings[string(i-1)][:,:,3] # For now take layer3
        amr.snt_tokens = snt_tokens[amr.id]
    end
end


function append_nodeembeddings_to_amrs(amrs)
     for amr in amrs
        try
            calculate_node_embeddings(amr)
        catch
            println("No alignment is found for ", amr.id)
        end
    end
end

""" Calculate node embeddings via alignments"""
function calculate_node_embeddings(amr)
    nodes_to_anchors = Dict()
    for align in amr.alignments
        nodeid = align["id"]
        node_aligned_tokens = [] 
        if haskey(align, "anchors")
            for anc in align["anchors"]
                push!(node_aligned_tokens, anc["#"])
            end
        end
        if haskey(align, "anchorings")
            for anc in align["anchorings"]
                for an in anc
                    push!(node_aligned_tokens, an["#"])
                end
            end
        end    
        nodes_to_anchors[nodeid] = node_aligned_tokens ## We can store node_to_anchors in amr field.
    end
    
    amr.nodes_to_anchors = nodes_to_anchors
    amr.nodeembeddings = zeros(length(amr.nodes),1024)

    for (node,anchors) in nodes_to_anchors
        for anc in anchors
            amr.nodeembeddings[node,:] =  amr.nodeembeddings[node,:] + amr.wordembeddings[:,anc]
        end
        # take avg. of word embeddings
        amr.nodeembeddings[node,:] /= length(anchors) 
    end
end



function calculate_distance_matrix(amr)    
    function calculate_direct_paths(amr)
        for (s,r,t) in amr.triplets
            amr.distance_matrix[s,t]=1
            amr.distance_matrix[t,s]=1
        end
    end

    function calculate_node_distances(amr, nodeid)
        #all_nodes = Dict()
        node_levels = Dict()
        ranges = collect(1:length(amr.nodes))

        for rang in ranges
            _nodes = []
            for (i,j) in enumerate(amr.distance_matrix[nodeid,:])
                if j == rang && i != nodeid
                    push!(_nodes, i)
                end
            end
            
            #node_levels[rang] = _nodes

            __nodes = []
            for nnode in _nodes
                for (i,j) in enumerate(amr.distance_matrix[nnode,:])
                    if j == 1 && i!=nodeid
                        push!(__nodes, i)
                        if amr.distance_matrix[nodeid,i] == 0 # not calculated
                            amr.distance_matrix[nodeid,i] =  rang + 1
                        end
                    end
                end
            end
            node_levels[rang+1] = __nodes
        end 
    end

    for i in 1:length(amr.nodes)
        calculate_direct_paths(amr)
        calculate_node_distances(amr,i)
    end
end

function append_distance_matrices(amrs)
    for amr in amrs
        calculate_distance_matrix(amr)
    end
end


## Main function

amrs = AMRReader(amr_file) ## 55898 amrs
alignments = get_alignments(alignment_file) ## 61341 alignments
embeddings = get_word_embeddings(embeddings_file) #55899 embeddings
snt_tokens = UdpipeReader(udpipe_file) # 122661, why?

append_alignments_to_amrs(amrs, alignments)
append_wordembeddings_to_amrs(amrs, snt_tokens, embeddings)
append_nodeembeddings_to_amrs(amrs)
append_distance_matrices(amrs)


probe = Probe(1024,1024)
initopt!(probe)
train(probe, amrs, 100)
