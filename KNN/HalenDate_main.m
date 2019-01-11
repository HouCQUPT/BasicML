%% KNN Halen Date 
% The data-set(filename:datingTestSet.txt) consisits of 1000 samples from
% each of three kinds of gentlemen who dating with Halen. Four features were
% measured from each sample:Frequent-flyer miles each year, percentage of
% time spent playing video games and Ice-cream consumed weekly in litres.
clc,close all,clear all;

%% Load data set 
data = load('datingTestSet.txt');
TestIndex = randperm(1000,100);     % get random number as test sample index
testdata = data(TestIndex,:);
data(TestIndex,:) = 0;            
traindata = data(data(:,4) ~= 0,:); % remaining sample as traindata  

%% Plot 
% Drawing 3-D scatter plot of train sample
[m,~] = size(traindata);
figure,
for ii = 1:m
    if (traindata(ii,4) == 1)
        C = 'r';
    elseif (traindata(ii,4) == 2)
        C = 'b';
    elseif (traindata(ii,4) == 3)
        C = 'g';
    end
    scatter3(traindata(ii,1),traindata(ii,2),traindata(ii,3),50,C,'filled'),
    hold on
end
title('3 Demensional Chart');
legend('Type1','Type2','Type3');

%% Data Normalization
% ---- Normalization Function: y=(x-MinValue)/(MaxValue-MinValue)
% normalization train data matrix
MinValueTrain = min(traindata(:,1:3));
MaxValueTrain = max(traindata(:,1:3));
MaxValueTrain = repmat(MaxValueTrain,900,1);
MinValueTrain = repmat(MinValueTrain,900,1);
traindata(:,1:3) = (traindata(:,1:3) - MinValueTrain) ./ (MaxValueTrain - MinValueTrain);    

% normalization test data matrix
MinValueTest = min(testdata(:,1:3));
MaxValueTest = max(testdata(:,1:3));
MaxValueTest = repmat(MaxValueTest,100,1);
MinValueTest = repmat(MinValueTest,100,1);
testdata(:,1:3) = (testdata(:,1:3) - MinValueTest) ./ (MaxValueTest - MinValueTest);       

%% Enclidean Distance
K = 5;
[MTrain,~] = size(traindata);
[MTest ,~] = size(testdata);
EDistance = zeros(MTrain,MTest);
for ii = 1:MTrain
    for jj = 1:MTest
        EDistance(ii,jj) = sqrt(sum((traindata(ii,1:3) - testdata(jj,1:3)).^2));
    end
end

% sort by min distance
[EDistance,index] = sort(EDistance);
K_EDistance = EDistance(1:K,:);     %   K nearest neighbors
K_index = index(1:K,:);             %   K nearest neighbors
K_type = zeros(K,MTest);
for ii = 1:MTest
    K_type(:,ii) = traindata(K_index(:,ii),4);
end

% caclulating K nearest total of type each test sample
TotalType = zeros(1,MTest);
type = zeros(1,3);
for ii = 1:MTest
    type(1,:) = 0;
    for jj = 1:K
        if K_type(jj,ii) == 1
            type(1,1) = type(1,1) + 1;
        elseif K_type(jj,ii) == 2
            type(1,2) = type(1,2) + 1;
        elseif K_type(jj,ii) == 3
            type(1,3) = type(1,3) + 1;
        end
    end
    [~,TotalType(ii)] = max(type);
end
error = zeros(MTest,1);
TotalType = TotalType';
error = testdata(:,4) - TotalType;
right = size(find(error == 0),1);
display(['错误个数:',num2str(MTest - right)]);
display(['正确率:',num2str(right ./ MTest)]);

            

        
