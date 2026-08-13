function [r, p] = MiniBall(data, inidx, bdidx)
    if isempty(inidx)
        if numel(bdidx) == 0
            r = 0;
            p = zeros(1, size(data, 2));
        elseif numel(bdidx) == 1
            r = 0;
            p = data(bdidx(1), :);
        elseif numel(bdidx) == 2
            r = norm(data(bdidx(1), :)-data(bdidx(2), :))/2;
            p = (data(bdidx(1), :)+data(bdidx(2), :))/2;
        elseif numel(bdidx) == 3
            [r, p] = CircumscribedCircleFor3points(data(bdidx(1), :), data(bdidx(2), :), data(bdidx(3), :));
        elseif numel(bdidx) == 4
            [p, r] = circumcenter(delaunayTriangulation(data(bdidx, :)));
        else
            error("unexpected dimension");
        end
    else
        [r, p] = MiniBall(data, inidx(2:end), bdidx);
        if norm(data(inidx(1), :) - p) > r
            [r, p] = MiniBall(data, inidx(2:end), unique([bdidx; inidx]));
        end
    end

end