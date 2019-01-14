%% Perceptron
clc,close all,clear all;

%% Load Linearly Separable Data-Set
load data.mat
load label.mat
plot(traindata(1:50,1),traindata(1:50,2),'+');
hold on
plot(traindata(51:150,1),traindata(51:150,2),'+');
title('Linearly Separable Data-set');legend('Positive Sample','Negative Sample');

%% Train
% Initilizing Para
weight = rand(2,1);     % weight vector
bias = 0.9;
eta = 0.1;                % learn step
temp = 1;
while temp ~= 0
    temp = 0;
    for ii = 1:size(traindata,1)
        if (label(ii,1) * ((traindata(ii,:) * weight) + bias))<= 0  % misclassification sample
            temp = temp + 1;
            %   update weight and bias
            weight = weight +  traindata(ii,:)' * label(ii,1);
            bias = bias +  label(ii,1);
        end
    end
end

hold on
% plot separating hyperplane
x = [4:8];
plot(x,-((weight(1,1) .* x) + bias) ./ weight(2,1),'-');