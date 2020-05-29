function report_uuas(preds, dataset)
    uuas_total = 0
    for (id, pred_distances) in preds
        amr = dataset[id]
        n = length(amr.al)
        gold_edges = union_find(n, pairs_to_distances(amr, amr.al_distances))
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
            union(i_index, j_index)
            push!(edges, (i_index, j_index)) 
        end
    end
    return edges
end
