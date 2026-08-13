function [r, p] = CircumscribedCircleFor3points(p1, p2, p3)
%     [BX, BY, BZ] = sphere;

    rs = NaN(3, 1);
    ps = NaN(3, size(p1, 2));


    i = 1;
    rs(i) = norm(p1 - p2)/2;
    ps(i, :) = (p1 + p2)/2;
%     surf(rs(i)*BX + ps(i, 1), rs(i)*BY + ps(i, 2), rs(i)*BZ + ps(i, 3), 'FaceAlpha', 0.01); hold on

    if norm(ps(i, :) - p3) > rs(i)
        rs(i) = NaN;
        ps(i, :) = NaN;
    end

    i = 2;
    rs(i) = norm(p2 - p3)/2;
    ps(i, :) = (p2 + p3)/2;
%     surf(rs(i)*BX + ps(i, 1), rs(i)*BY + ps(i, 2), rs(i)*BZ + ps(i, 3), 'FaceAlpha', 0.01);hold on

    if norm(ps(i, :) - p1) > rs(i)
        rs(i) = NaN;
        ps(i, :) = NaN;
    end

    i = 3;
    rs(i) = norm(p3 - p1)/2;
    ps(i, :) = (p3 + p1)/2;
%     surf(rs(i)*BX + ps(i, 1), rs(i)*BY + ps(i, 2), rs(i)*BZ + ps(i, 3), 'FaceAlpha', 0.01);hold on

    if norm(ps(i, :) - p2) > rs(i)
        rs(i) = NaN;
        ps(i, :) = NaN;
    end

    [r, ir] = min(rs);
    p = ps(ir, :);
    if isnan(r)
        a = p1 - p3;
        b = p2 - p3;
        na = norm(a); nb = norm(b); iab = dot(a, b);
        x = ( (na^2*nb^2 - nb^2*iab)*a + (na^2*nb^2 - na^2*iab)*b ) / (2*((na*nb)^2 - iab^2));
        p = x + p3;
        r = norm(x);
%         surf(r*BX + p(1, 1), r*BY + p(1, 2), r*BZ + p(1, 3), 'FaceAlpha', 0.1);hold on
    end
end