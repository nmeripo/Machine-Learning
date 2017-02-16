%Loading dataset
load 'faces.mat'

%Finding number of samples in test data and train data
test_data_n = size(testdata,1);
tr_data_n = size(traindata,1);


%Precomputing cosine Distance from test samples to training samples
for i = 1:test_data_n
    for k = 1:tr_data_n
        A(i, k) = cosineDistance(traindata(k,:),testdata(i,:));
    end 
end


%Precomputing cosine Distance from training samples to other training samples
for i = 1:tr_data_n
    for k = 1:tr_data_n
        C(i, k) = cosineDistance(traindata(k,:),traindata(i,:));
    end
end


%Intialize test_error and train_error vectors to store data to plot graph

test_error = [];
tr_error = [];
[t1 t2] = solution(1,A,C);
test_error = [test_error t1];
tr_error = [tr_error t2];

i = 10;

while i <= 100
    [t1 t2] = solution(i,A,C);
    test_error = [test_error t1];
    tr_error = [tr_error t2];
    i = i + 10;
end


x = linspace(0,100,11);
x((x == 0)) = 1;

%Plot graph                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
figure(1);
plot(x,tr_error,'r',x,test_error,'b');
title('Test Error and Train Error vs k neighbours')
legend('Train Error', 'Test Error','Location','southeast')
xlabel('k');
ylabel('Error');

