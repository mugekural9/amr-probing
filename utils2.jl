
## Util functions
function wrapmatrixh(matrix, maxlength)
    hdif = maxlength-size(matrix,2)
    h = zeros(size(matrix,1), hdif)
    tmpmatrix = zeros(size(matrix,1),maxlength)
    if hdif != 0; 
        tmpmatrix = hcat(matrix, h); 
    else 
        tmpmatrix = matrix
    end
    return tmpmatrix
end

function wrapmatrixv(matrix, maxlength)   

    vdif = maxlength-size(matrix,1)
    v = zeros(vdif, size(matrix,2))
    tmpmatrix = zeros(maxlength, size(matrix,2))
    if vdif != 0; 
        tmpmatrix = vcat(matrix,v); 
    else
        tmpmatrix = matrix
    end
    return tmpmatrix
end

function wrapmatrix(matrix, maxlength)   
    tmpmatrix = wrapmatrixh(matrix, maxlength)
    tmpmatrix = wrapmatrixv(tmpmatrix, maxlength)
    masks = zeros(size(tmpmatrix))
    masks[1:size(matrix,1), 1:size(matrix,2)] .= 1
    return tmpmatrix, masks
end
