clear;
clc;
profile on;
% Start timing
tic;
% % Check memory usage before
% memBefore = memory;
% % Example for GPU usage
% gpuDevice;
% % Load your dataset
data = readtable('Dataset.xlsx');

% Split the data into inputs (X) and outputs (Y)
X = data(:, 1:3);
Y = data(:, 4:end);

% Convert table to array if needed
X = table2array(X);
Y = table2array(Y);

% Determine the number of samples for training and testing
n = size(data, 1); % Total number of samples
numTest = floor(0.3 * n); % Number of test samples (30% of total)

% Separate the data into training and test sets (30% from the end for testing)
X_train = X(1:end-numTest, :); % 70% for training
Y_train = Y(1:end-numTest, :);

X_test = X(end-numTest+1:end, :); % 30% for testing
Y_test = Y(end-numTest+1:end, :);
% Define the neural network architecture
net = feedforwardnet([5, 5]); % Example architecture with 2 hidden layers of 10 neurons each
% Hyperparameters optimization 
% Grid search
hiddenLayerSize1 = 5:10:55;
hiddenLayerSize2 = 5:10:55;
hiddenLayerSize3 = 5:10:55;
learningRates = [0.01, 0.05, 0.1];
bestPerformance = inf;
i = 0;
% Define the list
metrics = {'mse'}; %, 'mae', 'mse', 'sae'};
for i = 1:length(metrics)
    metric = metrics{i};
    for h = hiddenLayerSize1
        for h2 = hiddenLayerSize2
            for h3 = hiddenLayerSize3
                for lr = learningRates
                    net = feedforwardnet([h h2 h3]);           
                    net.performFcn = metric; % Available options: 'mse', 'mae', 'sse' % Change the performance function to mean absolute error
                    net.trainParam.lr = lr; 
                    [net, tr] = train(net, X_train', Y_train');% Test the network
                    Y_pred = net(X_test');
                    performance = perform(net, Y_test', Y_pred); % Update the best performing network
                    mse = mean((Y_test' - Y_pred).^2);
                    mse = mean(mse);
                    rmse = sqrt(mse);
                    mae = mean(abs(Y_test' - Y_pred));
                    mae = mean(mae);
                    SStot = sum((Y_test' - mean(Y_test')).^2);
                    SSres = sum((Y_test' - Y_pred).^2);
                    r_squared = 1 - (SSres / SStot);
                    sse = sum((Y_test' - Y_pred).^2);
                    sse = sum(sse);
                    sae = sum(abs(Y_test' - Y_pred));
                    sae = sum(sae);
                    i = i+1;
                    store(i,:) = [h h2 h3 lr performance mse rmse mae r_squared sse sae];
                    if performance < bestPerformance
                        bestPerformance = performance;
                        bestNet = net;
                        bestHyperparameters = [h, h2, h3, lr];
                    end 
                end
            end
        end
    end
end
% Train the best model
net = bestNet;
[net, tr] = train(net, X_train', Y_train');
% End timing
elapsedTime = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsedTime);
% % Check memory usage after
% memAfter = memory;
% fprintf('Memory used: %.2f MB\n', (memAfter.MemUsedMATLAB - memBefore.MemUsedMATLAB) / 1e6);
% profile off;
% profile viewer;
% gpuDevice([]);