using StatsBase

function report_spearmanr(preds, dataset)
    five_to_fifty = []
    spamrs = Dict()
    for (id, pred_distances) in preds
        amr = dataset[id]
        gold_distances = amr.distance_matrix
        amrlength = length(amr.nodes) 
        if !(amrlength in keys(spamrs))
            spamrs[amrlength] = []
        end
        for i in 1:amrlength
            push!(spamrs[amrlength],corspearman(Array(pred_distances)[:,i], gold_distances[:,i]))
        end
    end
    mean_spamrs = mean.(values(spamrs))
    for (length, sp) in collect(zip(keys(spamrs), mean_spamrs))
      if 51>length>4
        push!(five_to_fifty, sp) 
      end
    end
    five_to_fifty_sprmean = mean(five_to_fifty) 
    return five_to_fifty_sprmean
end


function report_uuas_t(preds, amrs)
    uuas_total = 0
    for (id, pred_distances) in preds
        amr = amrs[id]
        n = length(amr.nodes)
        gold_edges = union_find(n, pairs_to_distances(amr, amr.distance_matrix))
        pred_edges = union_find(n, pairs_to_distances(amr, pred_distances))
        uuas_amr = length(findall(in(pred_edges), gold_edges)) / length(gold_edges)
        
        if isnan(uuas_amr); println(length(gold_edges)," n: $n"); end
        uuas_total += uuas_amr
    end
    return uuas_total / length(preds)
end



function report_uuas_t_upperbound(amrs)
    uuas_total = 0
    for amr in amrs
        n = length(amr.nodes)

        gold_edges = union_find(n, pairs_to_distances(amr, amr.distance_matrix))
        _gold_edges = union_find(n, pairs_to_distances(amr, amr.cleaned_distance_matrix))
        uuas_amr = length(findall(in(_gold_edges), gold_edges)) / length(gold_edges)

        if isnan(uuas_amr); println(length(gold_edges)," n: $n"); end
        uuas_total += uuas_amr
    end
    return uuas_total / length(amrs)
end


function report_uuas_g_upperbound(amrs)
    uuas_total = 0
    for amr in amrs
        n = length(amr.nodes)
        gold_edges = find_all_direct_edges(amr.distance_matrix)
        _gold_edges = find_all_direct_edges(amr.cleaned_distance_matrix)
        uuas_amr = find_overlapping_pairs(_gold_edges, gold_edges) / length(gold_edges)

        if isnan(uuas_amr); println(length(gold_edges)," n: $n"); end
        uuas_total += uuas_amr
    end
    return uuas_total / length(amrs)
end


function report_uuas_g(preds, amrs)
    uuas_total = 0
    for (id, pred_distances) in preds
        amr = amrs[id]
        n = length(amr.nodes)
        gold_edges = find_all_direct_edges(amr.distance_matrix)
        pred_edges =  span_graph(pred_distances) 
        uuas_amr = find_overlapping_pairs(pred_edges, gold_edges) / length(gold_edges)
        if isnan(uuas_amr); println(length(gold_edges)," n: $n"); end
        uuas_total += uuas_amr
    end
    return uuas_total / length(preds)
end


function span_graph(_distances)
    distances = deepcopy(_distances)
    for i in 1:size(distances,1)
        distances[i,i] = 100000000
    end    
    function connect_group(group)
        new_group = []
        out = length(group)
        usual_suspects = Dict()
        distances_to_suspects= Dict()
    
        for i in group
            node_usual_suspect = setdiff(sortperm(Array(distances[i,:]))[1:out], group)[:1]
            usual_suspects[i]=  node_usual_suspect
            distances_to_suspects[(i, usual_suspects[i])] = distances[i, node_usual_suspect]
        end
        closest_edge_idx = argmin(collect(values(distances_to_suspects)))
        closest_edge = collect(distances_to_suspects)[closest_edge_idx][1]
   
        for g in disc_comps
           if closest_edge[2] in g
               new_group = union(group,g)
           end
       end
       return new_group, closest_edge
    end
    
    missing_edges = Set()
    connected_graph = []
    disc_comps = disconnected_components(distances)
    group = first(disc_comps) # Start with the first one
    connected_graph = group
    while length(connected_graph) < size(distances,1) #number of nodes
        group, new_edge = connect_group(group)
        push!(missing_edges, new_edge)
        connected_graph = union(connected_graph, group)
    end

    for i in 1:size(distances,1)
        direct_edge =  (i,argmin(Array(distances[i,:])))
        if !((argmin(Array(distances[i,:])),i) in missing_edges)
            push!(missing_edges, direct_edge)
        end
    end
    return collect(missing_edges) #return as array
end



function disconnected_components(distances)
    ## Add direct edges
    edges = Dict() 
    for i in 1:size(distances,1) #number of nodes
        edges[i]= argmin(Array(distances[i,:]))
    end

    ## Add neighbors
    particles = Dict()
    neighbors = Dict()
    for (k,v) in edges
       if !haskey(neighbors, v) neighbors[v] = []; end
       push!(neighbors[v], k)  
    end

    for (k,v) in edges
        if !haskey(particles, k) particles[k] = []; end
        push!(particles[k], v)
        particles[k] = union(particles[k], neighbors[v])
    end

    ## Add neighbors of neighbors
    for (p,v) in particles
        for n in v 
            particles[p] = sort(union(particles[p], particles[n]))
            particles[n] = sort(union(particles[p], particles[n]))
        end
    end
    
    groups = union(values(particles))
    return groups
end



function find_all_direct_edges(distances)
    edges = []
    for i in 1:size(distances)[1]
       for j in 1:size(distances)[2]
             if distances[floor(Int,i),floor(Int,j)] == 1
                 if !((j,i) in edges)
                     push!(edges, (i,j))
                 end
             end
       end
    end
    return edges
end


function find_overlapping_pairs(pred_edges, gold_edges)
    overlap=0
    for i in 1:length(pred_edges)
        a,b = pred_edges[i]
        if (a,b) in gold_edges || (b,a) in gold_edges
            #println((a,b))
            overlap +=1
        end
    end
    return overlap
end

function pairs_to_distances(amr, distances)
    prs_to_distances =  Dict()
    for i in 1:size(distances,1)
        for j in 1:size(distances, 2)
            prs_to_distances[(i, j)] = distances[i,j]
        end
    end 
    return prs_to_distances
end


function union_find(n, prs_to_distances)
    function union(i, j)
        if findparent(i) != findparent(j)
            i_parent = findparent(i)
            parents[i_parent] = j
        end
    end
    function findparent(i)
        i_parent = i
        while true
            if i_parent != parents[i_parent]
                i_parent = parents[i_parent]
            else
                break
            end
        end
        return i_parent
    end

    edges = []
    local parents = collect(1:1:n)
    for ((i_index, j_index), distance) in sort(collect(prs_to_distances),by=x->x[2])
        i_parent = findparent(i_index)
        j_parent = findparent(j_index)
        if i_parent != j_parent
            #println("notsame parents of : ", i_index, "- ", j_index)
            union(i_index, j_index)
            push!(edges, (i_index, j_index)) 
        else
            #println("same parents of : ", i_index, "- ", j_index)
            #println("parents are: ", i_parent," -", j_parent)
        end
    end
    return edges
end




function mean_nodes(dataset)
    i=0
    for amr in dataset.amrs
        i += length(amr.nodes)
        end
    return i/length(dataset.amrs)
end



function mean_edges(dataset)
    i=0
    for amr in dataset.amrs
        i += length(amr.triplets)
        end
    return i/length(dataset.amrs)
end
