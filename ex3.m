clear all
close all
clc
%% input vector
X = [ 0,0;
      0,1;
      1,0;
      1,1]; 
%%  vector %% 
t = [0;
     1;
     1;
     0];

input_size  = 2;
Hidden_unit = 4;
output_unit = 1;
bias_v = 0.5;
bias_w = 0.5;
max = 200;


%% choosing the weights in range of -0.5 to 0.5
a = 0.5;
b = -0.5;         
V = (b-a).*rand(input_size,Hidden_unit) + a;
W = (b-a).*rand(output_unit,Hidden_unit) + a;
i = 1;

for iteration = 1:max
Error_value = 0;
for i = 1: 4 
%% forwarrd multiplication.

%% level1
Z_In = (X(i,:)*V) + bias_v;
z = 1.0 ./ (1.0 + exp(-Z_In));
z = z';

%% level2
Y_In = W*z + bias_w;
y = 1.0 ./ (1.0 + exp(-Y_In));

%% Backward propagation.%%
%% Find error
Error = t(i) - y;
alpha = 0.9;

%% level 3
Derivative_y = y'*(1-y);
Delta_y = Error * Derivative_y;
Delta_w = alpha * Delta_y .* z;
Delta_w_bias = alpha * Delta_y;

%% level 2
Derivative_z = z' *(1-z);
Delta_z = Derivative_z .* (W * Delta_y'); 
Delta_v = alpha* Delta_z' * X(i,:);
Delta_v_bias = alpha * sum(Delta_z);

%% update the weighs
W = (W' + Delta_w)' ;
bias_w = bias_w + Delta_w_bias;

V= V + Delta_v';
bias_v = bias_v + Delta_v_bias ;

%% incrementing loop
Error_value = Error_value + Error;
end
Error_plot(iteration) = abs(Error_value);
end 
plot(Error_plot);
