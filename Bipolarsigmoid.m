function g = Bipolarsigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = BIPOLAR SIGMOID(z) computes the sigmoid of z.
g = (1.0 - exp(-z)) ./ (1.0 + exp(-z));
end