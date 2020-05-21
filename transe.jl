include("data.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

mutable struct GraphElements
    e; l
end

function GraphElements(embed_dim::Int, num_entity::Int, num_rel::Int)
    k = 6/sqrt(embed_dim)
    e = param(rand(Uniform(-k, k), (embed_dim, num_entity)), atype=_atype)
    l = param(rand(Uniform(-k, k), (embed_dim, num_rel)), atype=_atype)
    #l = normalize_columns(l)
    GraphElements(e,l)
end

mutable struct triplet
    h; l; t 
end

function normalize_columns(m)
    ## Normalize m column vectors with L2 norm
    b = m' * m
    b = b[:,1]
    m = m ./ b'
    return m
end

function normalize_columns_v1(m)
    ## Normalize m column vectors with L1 norm
    #b = m[:,1]
    mags = sum(m, dims=1)
    m = m ./ mags
    return m
end

function loss_transe(ngbatch, gamma)
    loss = 0
    for (trp, corrupted_trp) in ngbatch
        pos_dis = g.e[:,trp.h] + g.l[:,trp.l] - g.e[:,trp.t]
        neg_dis = g.e[:,corrupted_trp.h] + g.l[:,corrupted_trp.l] - g.e[:,corrupted_trp.t]
        distance_diff = gamma + norm(pos_dis) - norm(neg_dis)
        loss += max(0,distance_diff)
    end
    return loss
end


function train(data, batchsize, gamma, lr)
    a = Iterators.Stateful(data)
    epoch = 1
    trainloss = 0
    while epoch <1000
        #g.e = normalize_columns(g.e)
        batch = collect(Iterators.take(a, batchsize))
        if !isempty(batch) 
            ngbatch = generate_negatives(batch)
            J = @diff loss_transe(ngbatch, gamma)
            trainloss += value(J)
            for par in params(g)
                g = grad(J, par)
                if isnothing(g) println("nothing"); return J, par; end
                update!(value(par), g, SGD(lr=lr))
            end
        else ## End of epoch
            a = Iterators.Stateful(data)
            #trainacc= "-"
            trainacc = accuracy_tail(data); 
            if trainacc == 1.0
                return;
            end
            println("epoch: $epoch, trainloss: $trainloss, trainacc: $trainacc")
            epoch += 1
            trainloss = 0
        end
    end
end

function generate_negatives(sbatch)
    trp_pairs = []
    for i in 1:length(sbatch)
        trp = sbatch[i]
        k = rand(1:length(rels))
        while k in rs[string(trp.h,"-",trp.l)]
            #println("once again because k: $k for $trp")
            k = rand(1:length(rels))
        end
        corrupted_trp = triplet(trp.h, trp.l, k)
        push!(trp_pairs, (trp, corrupted_trp))
    end
    return trp_pairs
end


function pred_tail(h,l)
    h_and_r =g.e[:,h] + g.l[:,l]
    trans = h_and_r .- g.e
    norms = []
    for i in 1:size(trans,2)
        if i!=h; push!(norms,norm(trans[:,i])); end
    end
    return argmin(norms) +1
end


function accuracy_tail(triplets)
    N = length(triplets)
    total_correct =  0
    for trp in triplets
        if trp.t in ranker(trp, 5)
            total_correct +=1
        end
    end
    println("total_correct: $total_correct, out of $N")
    return total_correct/N
end


function ranker(trp, topk)
    getrankid(rank) = return rank[1]
    getrank(trp) = return norm(g.e[:,trp.h] + g.l[:,trp.l] - g.e[:,trp.t])
    ranks = []
    for (label, id) in ents
        if id!=trp.h
           ctrp = triplet(trp.h, trp.l, id)
           push!(ranks, (id, getrank(ctrp)))
        end
    end
    sortedpreds =sort(ranks, by= x->x[2])
   return getrankid.(sortedpreds[1:min(length(sortedpreds),topk)])
end

