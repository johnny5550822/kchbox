% Computer numerical gradient

function numgrad = computeNumericalGradient(J, theta)

% Initialize numgrad with zeros
numgrad = zeros(size(theta));

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
size(theta);
e = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

%% ---------------------------------------------------------------
end
