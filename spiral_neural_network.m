clear;clc;
tic
rng(42)

%% load data
train_data = readmatrix('DATA_TRAIN.csv');
X_train = transpose(train_data(:,1:2));
Y0_train = transpose(train_data(:,3));
n_samples = length(Y0_train);

valid_data = readmatrix('DATA_VALID.csv');
X_valid = transpose(valid_data(:,1:2));
Y0_valid = transpose(valid_data(:,3));

%% Define network

% Set dimensions
n_input = 2;
n_hidden = 15;
L = 6; % number of layers including the output layer
n_output = 1;

% define weight initalizating functions
lower = -1;
upper = 1;
gen_weights_vanilla = @(in, out) (upper-lower).*rand(out, in) + lower;
gen_weights_xavier = @(in, out) gen_weights_vanilla(in,out) / sqrt(in);
gen_weights_xavier_norm = @(in, out) gen_weights_vanilla(in,out) / sqrt(6/(in + out));
gen_weights_kaiming = @(in, out) sqrt(2/in).*randn(out, in);

% choose weight init method
gen_weights = gen_weights_xavier_norm;

% initalize weights
W{1} = gen_weights(n_input + 1, n_hidden);
for i=2:L-1
    W{i} = gen_weights(n_hidden + 1, n_hidden + 1); 
end
W{L} = gen_weights(n_hidden + 1, n_output);

% Set an activation function for each layer
for i=1:L-1
    g{i} = @Tanh;
end
g{L} = @sigmoid;

%% Declare learning parameters & learn
eta      	= 7e-4;
n_epochs    = 400;
batch_size = 64;
lambda = 0.005; % l2 regularization param
alpha = 0; % momentum param
beta  = 0.99; % rmsProp param
decision_t = 0.5; % decision threshold
stop_early = 0; % if true, stops when validation accuracy reaches 100%

for ep = 1:n_epochs   
    samp_order = randperm(n_samples);
    % init cell array for momentum / rmsProp
    prev_grad = cell(length(W));
    grad_squared = cell(length(W));
    for i=1:L
        prev_grad{i} = zeros(size(W{i}, 1), size(W{i}, 2));
        grad_squared{i} = prev_grad{i};
    end
    for batch = 1:n_samples/batch_size
        % init d with same size as W
        d = cell(1,L);
        for i=1:L
            d{i} = zeros(size(W{i}, 1), size(W{i}, 2));
        end
        
        % pre allocate for speed
        x = cell(L+1);
        xp = cell(L+1);
        delta = cell(L);

        % Pick out random batch of examples
        batch_start = batch_size*(batch-1) + 1;
        batch_end = (batch_size)*batch;
        s    = samp_order(batch_start:batch_end);
        x{1} = X_train(:, s);
        y0   = Y0_train(s);
        

        % Forward pass
        for i=1:L-1
            [x{i+1}, xp{i+1}] = g{i}(W{i}*[x{i}; ones(1,size(x{i},2))]);
        end
        x{L+1} = g{L}(W{L}*x{L});

        % Backward pass        
        delta{L} = x{L+1}-y0;
        d{L} = d{L} + delta{L}*transpose(x{L});
        for i=1:L-1
            delta{L - i} = transpose(W{L-i+1})*delta{L-i+1}.*xp{L-i+1};
            d{L-i} = d{L - i} + delta{L-i}*transpose(x{L-i});
        end     
        % update weights
        for i=1:L
            grad = d{i}/batch_size + lambda*W{i};
            grad_squared{i} = beta * grad_squared{i} + (1-beta) * grad.^2;
            W{i} = W{i} -eta*grad./grad_squared{i}.^(1/2) + alpha*prev_grad{i};
            prev_grad{i} = -eta*grad; % save grad to use as momentum for next batch
        end
    end
    % calculate validation accuracy
    Y_valid = foward_pass(W,g,X_valid);
    Y_valid_pred = Y_valid > decision_t;
    ep_acc = nnz(Y_valid_pred == Y0_valid)/length(Y0_valid);
    acc(ep) = ep_acc;
    
    Y_train = foward_pass(W,g,X_train);
    Y_train_pred = Y_train > decision_t;
    ep_acc2 = nnz(Y_train_pred == Y0_train)/length(Y0_train);
    acc2(ep) = ep_acc2;
    
    % stop early to avoid overfitting
    if ep_acc == 1 && stop_early
        disp("stopped early after " + num2str(ep) + " epochs");
        break;
    end
end

%% evaluate network on validation dataset

% dense grid of points for creating decision boundary
range = -12:0.15:12;
[X,Y] = meshgrid(range, range);
X_boundary = transpose([X(:) Y(:)]);

Y_valid = foward_pass(W,g,X_valid);
Y_valid_pred = Y_valid > decision_t;

Y_boundary = foward_pass(W,g,X_boundary);
Y_boundary_pred = Y_boundary > decision_t;


%% plot predictions for validation

figure('Units','normalized','Position' ,[0.15,0.2,0.7,0.5]);
subplot(1,2,1);
plot_results(X_boundary, Y_boundary_pred, [240 217 14]/256, [96 224 16]/256);
plot_results(X_valid, Y_valid_pred, [219 24 24]/256, [0 94 255]/256);
title("Network performance on validation dataset");
subtitle("Accuracy: " + num2str(acc(end) * 100) + "%");

disp("Accuracy: " + num2str(acc(end)));
subplot(1,2,2);
plot(acc);
hold on;
plot(acc2);
title("accuracy over time");
xlabel("Epoch")
ylabel("Accuracy");
legend("validation", "training");

toc

function plot_results(X, Y_pred, c1, c2)
    spiral_1 = X(:, Y_pred);
    spiral_2 = X(:, ~Y_pred);
    scatter(spiral_1(1,:), spiral_1(2,:), 'MarkerFaceColor',c1, 'MarkerEdgeColor',c1);
    xlim([-12 12]);
    ylim([-12 12]);
    hold on;
    scatter(spiral_2(1,:), spiral_2(2,:), 'MarkerFaceColor',c2, 'MarkerEdgeColor',c2);
end

function Y = foward_pass(W, g, X)
    L = length(g) + 1;
    X_fp = cell(L);
    X_fp{1} = X;
    for i=2:L
        X_fp{i} = g{i-1}(W{i-1}*X_fp{i-1});
    end
    Y = X_fp{L};
end

function E = cross_entropy(p, y0)
    E = -(y0*log(p) + (1-y0)*log(1-p));
end

function [g, gp] = ReLU(x)
    g = (x>0).*x;
    gp = (x>0);
end

function [g, gp] = sigmoid(x)
    g = 1./(1+exp(-x));
    gp = g.*(1-g);
end

function [g, gp] = Tanh(x)
    g = tanh(x);
    gp = 1 - g.^2;
end

