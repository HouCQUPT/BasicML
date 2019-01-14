% MYSIGN Signum Function
% for each element of X. MYSIGN(X) return 1 if the element is grater than 
% and equal zero. -1 if the element is less than zero.
function y = mysign(x)
y = 1;
if x < 0
    y = -1;
end
end