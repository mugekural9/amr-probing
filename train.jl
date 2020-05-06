using Knet, LinearAlgebra
import Distributions: Uniform
include("mrpdata.jl")
include("transe.jl")
include("probe.jl")
include("data.jl")

## Overfitting to graph to learn AMR graph embeddings
function save_graph_embeddings(amr_dataset, probedim)
    global ents
    global rels
    global trps
    global rs
    global g

    for amr in amr_dataset.amrset
        id = amr.id
        try
            all_entities_dict= Dict()
            all_relations_dict = Dict()
            for e in amr.entities
                if !haskey(all_entities_dict, e)
                    all_entities_dict[e] = length(all_entities_dict)+1
                end
            end
            for r in amr.relations
                if !haskey(all_relations_dict, r)
                    all_relations_dict[r] = length(all_relations_dict)+1
                end
            end

            ents,  rels,  trps = all_entities_dict, all_relations_dict,  amr.triplets
            trips = []; rs = Dict()
            for trip in trps; push!(trips, triplet(ents[trip[1]], rels[trip[2]], ents[trip[3]])); end
            for tr in trips; key = string(tr.h, "-", tr.l); if !haskey(rs, key); rs[key]=[];end; if !(tr.t in rs[key]); push!(rs[key], tr.t); end; end

            g = GraphElements(probedim, length(ents), length(rels))
            batchsize= 1; gamma = 1; lr = 0.01
            train(trips, batchsize, gamma, lr)
            graph_emb = hcat(g.e, g.l)
            graph_emb = sum(graph_emb,dims=2)
            filename = string("resources/mrp/",amr.id,"_emb.jld2")
            @info "Saving graph embeddings to $filename..."
            Knet.save(filename, "graph_emb", graph_emb)
        catch
            @warn "Could not embed amr graph. $id"
        end
    end

end


function train(probe, probedim, embeddim, amrset)
    elmo_layer_no = 2
    epoch = 1
    lrr = 0.005
    trn = amrset[1:900]
    dev = amrset[900:1000]

    for epoch in 1:50
        trnloss = 0
        devloss = 0

        for (i, t) in enumerate(trn)
            sentid = i-1
            elmo_layers = embeddings[string(sentid)] # E×T×3
            embedding = elmo_layers[:,:, elmo_layer_no] # ExT
            sent_rep = sum(embedding, dims=2)
            graph_embedding_id = string("resources/mrp/", t.id,"_emb.jld2")
            if !isfile(graph_embedding_id); continue; end
            gold_graph_rep = Knet.load(graph_embedding_id, "graph_emb")
            J = @diff probe(sent_rep, gold_graph_rep)
            trnloss += value(J)
            for par in params(probe)
                g = grad(J, par)
                update!(value(par), g, Adam(lr=lrr))
            end
        end

        for (j, d) in enumerate(dev)
            j += 899
            sentid = j-1
            elmo_layers = embeddings[string(sentid)] # E×T×3
            embedding = elmo_layers[:,:, elmo_layer_no] # ExT
            sent_rep = sum(embedding, dims=2)
            graph_embedding_id = string("resources/mrp/",d.id,"_emb.jld2")
            if !isfile(graph_embedding_id); continue; end
            gold_graph_rep = Knet.load(graph_embedding_id, "graph_emb")
            devloss += probe(sent_rep, gold_graph_rep)
        end

        println("epoch: $epoch, trnloss: $trnloss, devloss: $devloss")
        # Reducing lr
        if epoch%2 == 0
            lrr /= 2
            println("lr reduced to $lrr")
        end
        epoch +=1
    end
end


probedim = 50; embeddim= 1024
amr_file = "data/mrp/2020/cf/training/amr.mrp"
amr_dataset = AMRReader(amr_file)
#save_graph_embeddings(amr_dataset, probedim)

probe = Probe(probedim, embeddim)
amrset = amr_dataset.amrset
train(probe, probedim, embeddim, amrset[1:1000])
