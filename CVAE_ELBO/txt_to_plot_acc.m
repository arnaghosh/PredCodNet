close all;clear;
files = ["log_CVAE01_MNIST_acc.txt","log_CVAE10_MNIST_acc.txt","log_CVAE11_MNIST_acc.txt", ...
    "log_CVAE_Ztfo1_MNIST_acc.txt","log_CVAE_Ztfo2_MNIST_acc.txt"];
fractionLabelsArr = [];
bestAccuracyArr = [];
for filename=files
    fileID = fopen(filename,'r');
    fractionLabels = [];
    bestAccuracy = [];
    fline = fgetl(fileID);
    fline = fgetl(fileID);
    count = 0;
    while ~feof(fileID)
        if contains(fline,'best')
            percent_ind = strfind(fline,'%');
            fraction = sscanf(fline(strfind(fline,'For ')+4:percent_ind(1)),'%f');
            acc = sscanf(fline(strfind(fline,'is ')+3:percent_ind(2)),'%f');
            count=count+1;
            fractionLabels(count) = fraction;
            bestAccuracy(count) = acc;
        end
        fline = fgetl(fileID);
    end
    fractionLabelsArr = cat(1,fractionLabelsArr,fractionLabels);
    bestAccuracyArr = cat(1,bestAccuracyArr,bestAccuracy);
end
for i=linspace(1,size(bestAccuracyArr,1),size(bestAccuracyArr,1))
    bestAccuracyArr(i,:) = 100*bestAccuracyArr(i,:)/bestAccuracyArr(i,end);
end
figure(1)
plot(fractionLabelsArr',bestAccuracyArr','Linewidth',2);
set(gca,'fontsize',18)
legend('CVAE01','CVAE10','CVAE11','CVAE tfo1','CVAE tfo2');
xlabel('Batches');
ylabel('Classification accuracy (% of accuracy with all labels)');
title('Digit classification accuracy with limited labels for different CVAE architectures')