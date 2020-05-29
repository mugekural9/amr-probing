include("probe.jl")
include("data.jl")
include("report.jl")

elmo_layer = 1
amr_file = "data/mrp/2019/training/amr/wsj.mrp"
amrset = AMRReader(amr_file, elmo_layer).amrset

function initopt!(probe::Probe, optimizer="Adam()")
    for par in params(probe)
        par.opt = eval(Meta.parse(optimizer))
    end
end

function train(probe, amrset, numepoch)
    for epoch in 1:numepoch
        trnloss = 0
        devloss = 0
        ## Trn
        for amr in amrset[1:80]
            J = @diff probe_distance(probe, amr.al_embs, amr.al_distances)
            trnloss += value(J)
            for p in params(probe)
                g = grad(J, p)
                update!(value(p), g, p.opt)
            end
        end
        ## Dev 
        i = 1
        devpreds = Dict()
        for amr in amrset[80:end]
            dists, J = pred_distance(probe, amr.al_embs, amr.al_distances)
            devloss += J
            devpreds[79+i] = dists
            i += 1
        end
        uuas = report_uuas(devpreds, amrset)
        println("epoch: $epoch, trnloss: $trnloss, devloss: $devloss, dev uuas: $uuas")
        epoch +=1

        # Reducing lr 
        if epoch%10 != 0; continue; end
        lrr = Any
        for p in params(probe)
            p.opt.lr =  p.opt.lr/2
            lrr = p.opt.lr
        end
        println("lr reduced to $lrr")
    end
end


probe = Probe(1024,1024)
initopt!(probe)
train(probe, amrset, 100)
