%% K Nearest Neighbor
% ----- Iris Data Set
clc,clear all, close all;
%% Load Irit Data
data = load('IrisData.txt');
data = data(:,2:6);
% ---- train data ----
train_data = [data(1:40,:);data(51:90,:);data(101:140,:)];
train_label = train_data(:,5);
train_data = train_data(:,1:4);

% ---- test data ----
test_data = [data(41:50,:);data(91:100,:);data(141:150,:)];
test_label = test_data(:,5);
test_data = test_data(:,1:4);

%% K Nearest Neighbor
% ----- Calculating the eucliden distance of the train vector from test
% ----- vector
[Ntrain,~] = size(train_data);
[Ntest, ~] = size(test_data);
Distance   = zeros(Ntrain,Ntest);
for ii = 1:Ntrain
    for jj = 1:Ntest
        % Eucliden Distance
        Distance(ii,jj) = sqrt(sum((train_data(ii,:) - test_data(jj,:)).^2));
    end
end

% ---- Calculating mean distance of each type from test sample to train sample
type1 = Distance(1:40,:);       % Iris Setosa
type2 = Distance(41:80,:);      % Iris Versicolour
type3 = Distance(81:120,:);     % Iris Virginica
meantype1 = mean(type1);
meantype2 = mean(type2);
meantype3 = mean(type3);
meandistance = [meantype1;meantype2;meantype3];
[~,PreType] = min(meandistance);       % PreType is Iris type predicted by KNN
error = 0;
total = 0;
PreType = PreType';
error = abs(sum(PreType - test_label));
[total,~] = size(test_label);       % number of test data
display(['测试个数:',num2str(total)]);
display(['错误个数:',num2str(error)]);
display(['正确率：',num2str((total-error) ./ total)]);


