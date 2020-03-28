struct AMR
    id
    snt
end

## Text Reader
struct AMRReader
    file::String
    amrset
    ninstances::Int64
end

function AMRReader(file)
    amrset = []
    i = 0
    state = open(file);
    while true
        if eof(state)
            close(state)
            break
        else
            i +=1 
            line = strip(readline(state))
            idpart   = findfirst("::id",line)
            datepart = findfirst("::date",line)
            sentpart = findfirst("::snt",line)
            if !(isnothing(idpart))
                istrt, iend = idpart
                dstrt, dend = datepart
                id = line[iend+1: dstrt-1]
                line2 = strip(readline(state))
                sentpart = findfirst("::snt",line2)
                if !(isnothing(sentpart))
                    sstrt, send = sentpart
                    sent = line2[send+1:end]
                    push!(amrset, AMR(id,sent))
                end
            end
        end
    end
    AMRReader(file, amrset, length(amrset))
end

datadir = "data/abstract_meaning_representation_amr_2.0/data/amrs/unsplit"
boltfile = "$datadir/amr-release-2.0-amrs-bolt.txt"
bolt = AMRReader(boltfile)