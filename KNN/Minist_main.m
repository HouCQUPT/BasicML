%% KNN Minist
% ---- author: Wei How
clc, clear all;

%% Loading Minist Data
traindata = load('train.mat');      traindata   = traindata.data;
trainlabel = load('trainlabel.mat');trainlabel  = trainlabel.trainLabel;
testdata = load('test.mat');        testdata    = testdata.data;
testlabel = load('testlabel.mat');  testlabel   = testlabel.label;

%% K-Nearest Neighbor
[mTrain,~] = size(traindata);
[mTest,~] = size(testdata);
K = 5;         % parameter
EDistance = zeros(mTrain, mTest);   % Eucliden Distance
for ii = 1:mTrain
    for jj = 1:mTest
        EDistance(ii,jj) = sqrt(sum((traindata(ii,:) - testdata(jj,:)).^2));
    end
end

% sort
[EDistance, index] = sort(EDistance);
k_EDistance = EDistance(1:K,:);
k_index = index(1:K,:);
k_type = zeros(K+1, mTest);        % store k-nearest sample of type
for ii = 1:mTest
    k_type(1:K,ii) = trainlabel(k_index(:,ii),1);
    
    % tabulate(X£©takes a vector X and returns a matrix
    % input 'help tabulate' in command line to get more help
    table = tabulate(k_type(1:K,ii)'); 
    [maxCount,idx] = max(table(:,2));
    k_type(K+1,ii) = table(idx);
    
end

result = k_type(K+1,:)' - testlabel;
right = size(find(result == 0),1);       % number of right sample by KNN classifier
display(['----------------------- K:', num2str(K)]);
display(['----------- Error number :',num2str(mTest - right)]);
display(['-- Classitication rate   :',num2str(right ./ mTest)]);