% Calculate sigmoid gradient

function g = sigmoidGradient(z)

g = zeros(size(z));
g = sigmoid(z).*(1-sigmoid(z));

end
