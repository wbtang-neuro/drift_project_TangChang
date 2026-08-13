function [Persistence, Rinfs] = get_PD_H01_from_2Ddata(data)
    if size(data, 2) ~= 2
        error("data should be N x 2 Matrix.");
    end

    % "Calculating Delaunay Triangulation ..."
    Delaunay_SF = delaunayTriangulation(data);
    Delaunay_VS = (1:size(data, 1))';
    Delaunay_ED = edges(Delaunay_SF);
    Delaunay_SF = sort(Delaunay_SF.ConnectivityList, 2);
    
    nSim_0 = size(Delaunay_VS, 1);
    nSim_01 = size(Delaunay_VS, 1) + size(Delaunay_ED, 1);
    nSim_012 = size(Delaunay_VS, 1) + size(Delaunay_ED, 1) + size(Delaunay_SF, 1) ;
    
    % "Calculating Boundary Oparator Matrix ..."
    B = zeros(nSim_012);
    parfor is = 1:size(Delaunay_ED, 1)
    % for is = 1:size(Delaunay_ED, 1)
        Bcol = zeros(nSim_012, 1);
        for bd = nchoosek(Delaunay_ED(is, :), 1)'
            ibd = find(ismember(Delaunay_VS, bd', "rows"));
            Bcol(ibd, 1) = 1;
        end
        B(:, is + nSim_0) = Bcol;
    end
    parfor is = 1:size(Delaunay_SF, 1)
    % for is = 1:size(Delaunay_SF, 1)
        Bcol = zeros(nSim_012, 1);
        for bd = nchoosek(Delaunay_SF(is, :), 2)'
            ibd = find(ismember(Delaunay_ED, bd', "rows")) + nSim_0;
            Bcol(ibd, 1) = 1;
        end
        B(:, is + nSim_01) = Bcol;
    end

    
    % "Calculating Minimum Radius of Bounding Shphre for Each Sets of Points ..."
    Rmin_SF = NaN(1, size(Delaunay_SF, 1));
    for is = 1:size(Delaunay_SF, 1)
        Rmin_SF(1, is) = MiniBall(data(Delaunay_SF(is, :), :), (1:3)', []);
    end
    Rmin_ED = NaN(1, size(Delaunay_ED, 1));
    for is = 1:size(Delaunay_ED, 1)
        Rmin_ED(1, is) = MiniBall(data(Delaunay_ED(is, :), :), (1:2)', []);
    end
    Rmin_VS = NaN(1, size(Delaunay_VS, 1));
    for is = 1:size(Delaunay_VS, 1)
        Rmin_VS(1, is) = MiniBall(data(Delaunay_VS(is, :), :), [1], []);
    end
    Rs = unique([Rmin_VS Rmin_ED Rmin_SF]);

    
    
    % "Calculating Filtration of Alpha Simplicail Complexes ..."
    count = 1;
    Alpha_index = NaN(1, nSim_012);
    Alpha_Rs = NaN(1, nSim_012);
    Alpha_dims = NaN(1, nSim_012);
    for ir = 2:numel(Rs)
        VS_index = find((Rs(ir-1) <= Rmin_VS) & (Rmin_VS < Rs(ir)));
        ED_index = nSim_0 + find((Rs(ir-1) <= Rmin_ED) & (Rmin_ED < Rs(ir)));
        SF_index = nSim_01 + find((Rs(ir-1) <= Rmin_SF) & (Rmin_SF < Rs(ir)));
        ccindex = [VS_index ED_index SF_index];
        dims = [0*ones(size(VS_index)) 1*ones(size(ED_index)) 2*ones(size(SF_index))];
        Alpha_index(1, count:(count+numel(ccindex))-1) = ccindex;
        Alpha_Rs(1, count:(count+numel(ccindex))-1) = Rs(ir-1);
        Alpha_dims(1, count:(count+numel(ccindex))-1) = dims;
        count = count+numel(ccindex);
    end
    VS_index = find( Rmin_VS == Rs(ir));
    ED_index = nSim_0 + find(Rmin_ED == Rs(ir));
    SF_index = nSim_01 + find(Rmin_SF == Rs(ir));
    ccindex = [VS_index ED_index SF_index];
    dims = [0*ones(size(VS_index)) 1*ones(size(ED_index)) 2*ones(size(SF_index))];
    Alpha_index(1, count:(count+numel(ccindex))-1) = ccindex;
    Alpha_dims(1, count:(count+numel(ccindex))-1) = dims;
    Alpha_Rs(1, count:(count+numel(ccindex))-1) = Rs(ir);
    count = count+numel(ccindex)-1;
    
    Alpha_index = Alpha_index(1, 1:count);
    Alpha_dims = Alpha_dims(1, 1:count);
    Alpha_Rs = Alpha_Rs(1, 1:count);
    % "--Having "+count+" Simplicies--"
    
    
    
    
    % "Sorting Boundary Oparator Matrix by Filtration ..."
    B_alpha = B(Alpha_index, Alpha_index);
    
    % "Reducing Boundary Oparator Matrix ..."
    [B_alpha_reduced_Sparse, SV] = Left2RightReduction(B_alpha);

    
    
    lowjs = NaN(1, size(B_alpha_reduced_Sparse, 2));
    for j = 1:size(B_alpha_reduced_Sparse, 2)
        if ~isempty(find(B_alpha_reduced_Sparse(:, j), 1, "last"))
            lowjs(1, j) = find(B_alpha_reduced_Sparse(:, j), 1, "last");
        end
    end
    lowjs_test = sort(lowjs(~isnan(lowjs)));
    lowjs_test = lowjs_test(1, 2:end) - lowjs_test(1, 1:end-1);
    if any(lowjs_test == 0)
        error("Matrix was not Reduced completely");
    else
        % "Matrix was Reduced propery"
    end
    
    % "Getting Persistent Diagram ..."
    Persistence = {[], []};
    Rinfs = zeros(1, numel(Persistence));
    for j = 1:size(B_alpha_reduced_Sparse, 2)
        if all(B_alpha_reduced_Sparse(:, j) == 0)
            % "sigma_j is positive";
            p = unique(Alpha_dims(1, full(SV(:, j))==1)) + 1;
            k = find(lowjs == j, 1);
            if isempty(k)
                Persistence{p} = [Persistence{p}; [Alpha_Rs(1, j), inf]];
            end
        else
            % "sigma_j is negative";
            p = unique(Alpha_dims(1, full(SV(:, j))==1));
            Persistence{p} = [Persistence{p}; [Alpha_Rs(1, lowjs(1, j)), Alpha_Rs(1, j)]];
            if Alpha_Rs(1, j) > Rinfs(1, p)
                Rinfs(1, p) = Alpha_Rs(1, j);
            end
        end
    end
end
