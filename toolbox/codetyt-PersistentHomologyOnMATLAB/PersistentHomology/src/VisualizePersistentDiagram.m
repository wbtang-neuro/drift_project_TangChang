function VisualizePersistentDiagram(PD, Rinfs)
    colors = ["blue", "red", "green"];
    labels = ["H_0", "H_1", "H_2"];

    figure;
    for iPD = 1:numel(PD)
        subplot(numel(PD), 1, iPD);
        points = unique(PD{iPD}, 'row');
        scatter(points(:, 1), points(:, 2), 10, colors(1, iPD), "filled"); hold on
        points_inf = points(isinf(points(:, 2)), :);
        points_inf(isinf(points_inf)) = Rinfs(1, iPD);
        scatter(points_inf(:, 1), points_inf(:, 2), 30, colors(1, iPD), 'x'); hold on

        plot(0:Rinfs(1, iPD)*1.2, 0:Rinfs(1, iPD)*1.2, '--k'); hold on
        
        title(labels(1, iPD));
        pbaspect([1 1 1]);
    end
end