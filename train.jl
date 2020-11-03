include("report.jl")

function initopt!(probe::Probe, optimizer="Adam()")
    for par in params(probe)
        par.opt = eval(Meta.parse(optimizer))
    end
end

function train(probe, amrs, numepoch)
    for epoch in 1:numepoch
        println("epoch: $epoch")
        trnloss = 0
        devloss = 0

        ## Trn
        for amr in amrs[1:12000]
            if amr.nodeembeddings != Any && amr.distance_matrix != Any
                J = @diff probe_distance(probe, amr.nodeembeddings, amr.distance_matrix)
                trnloss += value(J)
                for p in params(probe)
                    g = grad(J, p)
                    update!(value(p), g, p.opt)
                end
            else
                println("amr is excluded from training: ", amr.id)
            end
        end
        println("trnloss: $trnloss")        

        ## Dev
        from, to = 12000, 15000
        offset = from-1 
        devpreds = Dict()
        for (i,amr) in enumerate(amrs[from:to])
            if amr.nodeembeddings != Any && amr.distance_matrix != Any
                dists, J = pred_distance(probe, amr.nodeembeddings, amr.distance_matrix)
                devloss += J
                devpreds[i+offset] = dists
            else
                println("amr is excluded from dev: ", amr.id)
            end
        end
        dev_uuas = report_uuas(devpreds, amrs)
        println("dev loss: $devloss")
        println("dev uuas: $dev_uuas")
        
        ## Reducing lr at every 10 epochs
        if epoch%10 == 0;
            lrr = Any
            for p in params(probe)
                p.opt.lr =  p.opt.lr/2
                lrr = p.opt.lr
            end
            println("lr reduced to $lrr")
        end
        
        epoch +=1        
    end
end

