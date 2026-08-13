function [SBSV, SV] = Left2RightReductionSparse_V2(B)
    nsimplex = size(B, 1);
    SB = sparse(B);
    SV = sparse(eye(nsimplex));

    SBSV = mod(SB*SV, 2);
    lowjs = false(nsimplex); %js s.t. lowj == i in SBSV
    parfor j = 1:nsimplex
        lowjs_col = false(nsimplex, 1);
        lowj = find(SBSV(:, j) == 1, 1, "last");
        if ~isempty(lowj)
            lowjs_col(lowj, 1) = true
        end
        lowjs(:, j) = lowjs_col;
    end

    inzcols = any(lowjs, 1); %index of nonzero columns in SBSV

    for j = find(inzcols)
%         j
        lowj = find(lowjs(:, j), 1);
        jj = find(lowjs(lowj, 1:j-1), 1);
        while ~isempty(jj)
%             SV(:, j) = SV*sparse(ColAdd(jj, j, 1, nsimplex));
            SV(:, j) = mod(SV(:, j) + SV(:, jj), 2);
            SBSV(:, j) = mod(SBSV(:, j) + SBSV(:, jj), 2);
            lowjs(lowj, j) = false;
            lowj = find(SBSV(:, j), 1, "last");
            if ~isempty(lowj)
                lowjs(lowj, j) = true;
            end
            jj = find(lowjs(lowj, 1:j-1), 1);
        end
    end 
end
