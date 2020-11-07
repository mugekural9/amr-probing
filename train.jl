include("report.jl")
include("utils2.jl")

using  IterTools
import Base: length, iterate

mutable struct Dataset
    amrs::Array{Any,1}
    batchsize
    ids
end

function Dataset(amrs, batchsize)
    damrs =  sort(collect(amrs),by=x->length(x.nodes), rev=true)
    i(x) = return x.id
    ids = i.(damrs)
    Dataset(damrs, batchsize, ids)
end

function al(x) 
  return length(x.nodes)
end

function iterate(d::Dataset, state=collect(1:length(d.amrs)))
   
    #println("hey! ", length(state))
    new_state = copy(state)
    new_state_len = length(new_state) 
    if new_state_len == 0 
        return nothing 
    end
    max_ind = min(new_state_len, d.batchsize)  
    amrs = d.amrs[new_state[1:max_ind]]
    amrlengths = al.(amrs)
    embeddim = size(amrs[1].nodeembeddings,1)
    maxlength = amrlengths[1]
    minim = min(d.batchsize, length(amrs))
    batch = []
    masks = []
    #golddepths = []
    golddistances = []

    for b in 1:minim
      push!(batch, wrapmatrixh(amrs[b].nodeembeddings', maxlength))
      gold, mask = wrapmatrix(amrs[b].distance_matrix, maxlength)
      #gold_depth = wrapmatrixv(sents[b].depths, maxlength)

      push!(golddistances, gold)
      #push!(golddepths, gold_depth)
      push!(masks, mask)
    end

    batch = cat(batch..., dims=3)
    #golddepths = cat(golddepths..., dims=3)
    golddistances = cat(golddistances..., dims=3)
    masks = cat(masks..., dims=3)
    deleteat!(new_state, 1:max_ind)
    return ((batch, golddistances, masks, amrlengths), new_state)
end

function length(d::Dataset)
    d, r = divrem(length(d.amrs), d.batchsize)
    return r == 0 ? d : d+1
end


# function initopt!(probe::Probe, optimizer="SGD()")
#     for par in params(probe)
#         par.opt = eval(Meta.parse(optimizer))
#     end
# end

# function train(probe, amrs, numepoch)
#     for epoch in 1:numepoch
#         println("epoch: $epoch")
#         trnloss = 0
#         devloss = 0

#         ## Trn
#         for (i,amr) in enumerate(amrs[1:19000])
#             if !amr.has_missing_alignment && amr.nodeembeddings != Any && amr.distance_matrix != Any
#                 J = @diff probe_distance(probe, amr.nodeembeddings, amr.distance_matrix)
               
#                 trnloss += value(J)
#                 for p in params(probe)
#                     g = grad(J, p)
#                     update!(value(p), g, p.opt)
#                 end
#             else
#                 println("amr is excluded from training: ", amr.id)
#             end
#         end
#         println("trnloss: $trnloss")        

#         ## Dev
#         from, to = 19000, 20000
#         offset = from-1 
#         devpreds = Dict()
#         for (i,amr) in enumerate(amrs[from:to])
#             if !amr.has_missing_alignment && amr.nodeembeddings != Any && amr.distance_matrix != Any
#                 dists, J = pred_distance(probe, amr.nodeembeddings, amr.distance_matrix)
#                 devloss += J
#                 devpreds[i+offset] = dists
#             else
#                 println("amr is excluded from dev: ", amr.id)
#             end
#         end
#         dev_uuas = report_uuas(devpreds, amrs)
#         println("dev loss: $devloss")
#         println("dev uuas: $dev_uuas")
        
#         # ## Reducing lr at every 10 epochs
#         # if epoch%10 == 0;
#         #     lrr = Any
#         #     for p in params(probe)
#         #         p.opt.lr =  p.opt.lr/2
#         #         lrr = p.opt.lr
#         #     end
#         #     println("lr reduced to $lrr")
#         # end
        
#         epoch +=1        
#     end
# end

# function setlr(lr)
#     for p in params(probe)
#         p.opt.lr = lr
#     end
# end


function loss_distance(probe, data)
    loss = 0
    for (batch, golddistances, masks, amrlengths) in data
        loss  += probetransform(probe, batch, golddistances,  masks, amrlengths)
    end
    return loss
end

function train_distances(epochs, probe, trn, dev)
    best = 1000000000; patience = 3; num_bad_epochs = 0    
    trnbatches = collect(trn)
    devbatches = collect(dev)
    
    epoch = adam(probetransform, ((probe, batch, golds, masks, amrlengths) for (batch, golds, masks, amrlengths) in trnbatches))

    for e in 1:epochs
      progress!(epoch) 
      trnloss = loss_distance(probe, trnbatches)
      devloss = loss_distance(probe, devbatches)
      println("epoch $e, trnloss: $trnloss, devloss: $devloss")

      # Reducing lr
      if devloss < best
            best = devloss
            num_bad_epochs = 0
      else
          num_bad_epochs +=1
      end
      if num_bad_epochs == patience
          lrr = Any
          for p in params(probe)
              p.opt.lr =  p.opt.lr/2
              lrr = p.opt.lr
          end
          println("lr reduced to $lrr")
          num_bad_epochs = 0
      end    


        if e % 5 == 0
        # TODO refactor here
        devpreds = Dict()
        for (k, (batch, golds, masks, amrlengths)) in enumerate(devbatches[1:end-1])
            dpreds, _ = pred_distance(probe, batch, golds, masks, amrlengths)
            #println("dpreds: ", dpreds) 
            id = 4*k -3
            devpreds[id]   = dpreds[:,:,1][1:amrlengths[1],1:amrlengths[1]]
            devpreds[id+1] = dpreds[:,:,2][1:amrlengths[2],1:amrlengths[2]]
            devpreds[id+2] = dpreds[:,:,3][1:amrlengths[3],1:amrlengths[3]]
            devpreds[id+3] = dpreds[:,:,4][1:amrlengths[4],1:amrlengths[4]]
            k += 1
        end
        five_to_fifty_sprmean = report_spearmanr(devpreds, dev.amrs)
        uuas = report_uuas(devpreds, dev.amrs)
        println("dev 5-50 spearman mean: $five_to_fifty_sprmean, uuas: $uuas")
        
        ## Saving the probe
        #probename = "probe_rank1024_elmo_v1.jld2"
        #@info "Saving the probe $probename" 
        #Knet.save(probename,"probe",probe)
    end
 end
end

