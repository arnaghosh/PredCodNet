close all;clear;
files = ["log_CVAE01_MNIST.txt","log_CVAE10_MNIST.txt","log_CVAE11_MNIST.txt", ...
    "log_CVAE_Ztfo1_MNIST.txt","log_CVAE_Ztfo2_MNIST.txt"];
trainLossArr = [];
testLossArr = [];
for filename=files
    fileID = fopen(filename,'r');
    trainLoss = [];
    testLoss = [];
    fline = fgetl(fileID);
    fline = fgetl(fileID);
    train_cnt = 0;
    test_cnt = 0;
    while ~feof(fileID)
        if contains(fline,'Test')
            loss = sscanf(fline(strfind(fline,'loss')+6:end),'%f');
            test_cnt=test_cnt+1;
            testLoss(test_cnt) = loss;
        elseif contains(fline,'Train')
            loss = sscanf(fline(strfind(fline,'Loss')+6:end),'%f');
            train_cnt=train_cnt+1;
            trainLoss(train_cnt) = loss;
        end
        fline = fgetl(fileID);
    end
    trainLossArr = cat(1,trainLossArr,trainLoss);
    testLossArr = cat(1,testLossArr,testLoss);
end
figure(1)
plot(200*linspace(1,train_cnt,train_cnt),trainLossArr','Linewidth',3);
set(gca,'fontsize',18)
legend('CVAE01','CVAE10','CVAE11','CVAE tfo1','CVAE tfo2');
xlabel('Batches');
ylabel('Train Loss');
title('Training Loss curves for different CVAE architectures')
figure(2)
plot(5*linspace(1,test_cnt,test_cnt),testLossArr','Linewidth',3);
set(gca,'fontsize',18)
legend('CVAE01','CVAE10','CVAE11','CVAE tfo1','CVAE tfo2');
xlabel('Epochs');
ylabel('Test Loss');
title('Testing Loss curves for different CVAE architectures')