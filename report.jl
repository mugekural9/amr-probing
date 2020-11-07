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

function report_uuas(preds, amrs)
    uuas_total = 0
    for (id, pred_distances) in preds
        amr = amrs[id]
        n = length(amr.nodes)
        gold_edges = union_find(n, pairs_to_distances(amr, amr.distance_matrix))
        pred_edges = union_find(n, pairs_to_distances(amr, pred_distances))
        uuas_amr = length(findall(in(pred_edges), gold_edges)) / length(gold_edges)
        #println("gold_edges: $gold_edges")
        #println("pred_edges: $pred_edges")

        if isnan(uuas_amr); println(length(gold_edges)," n: $n"); end
        uuas_total += uuas_amr
    end
    return uuas_total / length(preds)
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

function disconnected_components(distances)
    edges = Dict() 
    for i in 1:size(distances,1) # it will be square matrix
           edges[i]= argmin(Array(distances[i,:]))
    end
    
    big_particles = Dict()
    particles = Dict()
    for (k,v) in edges
        particle = []
        append!(particle, [k, v])
        if edges[v] != k
            append!(particle, edges[v])
        end

        big_particle = []
        for p in particle
            if !haskey(particles,p)
                particles[p] = particle
            else
                particles[p] = union(particles[p], particle)
            end        
            big_particle = union(big_particle, particles[p])
        end
        
        for p in particle
            big_particles[p] = big_particle
        end
    end

    groups = Set()
    for v in values(big_particles)
        maxparticle = []
        for j in v     
            if length(maxparticle) < length(big_particles[j])
                maxparticle= sort(big_particles[j])
           end
        end
            push!(groups, maxparticle)
    end
    return groups
end
