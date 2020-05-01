file = "$datadir/amr-release-2.0-amrs-bolt.txt"
state = open(file);
open("bolt-sentences.txt", "w") do io
    while true
        if eof(state)
            close(state)
            break
        else
            line = strip(readline(state))
            sentpart = findfirst("::snt",line)
            if !isnothing(sentpart)
                sent = string(strip(line[sentpart.stop+1:end]), "\n")
                write(io, sent)
            end
        end
    end
end;

