function [output] = list2vec(A,times)
output = zeros(size(times));
for i=1:length(A(:,1)),
    sp = find(times>=A(i,1),1,'first');
    ep = find(times<=A(i,2),1,'last');
    output(sp:ep) = 1;
end