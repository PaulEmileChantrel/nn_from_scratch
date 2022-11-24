function[]=NeuralNetwork(numI,numH,numO)

numI=784;
numH=[40 20];
numO=10;


nHL=length(numH);


input=loadMNISTImages('train-images.idx3-ubyte');
result=loadMNISTLabels('train-labels.idx1-ubyte');
results=labels2Array(result);

%show1Im(input(:,1))

% size(result)        %60000 X 1
% length(result)      %60000
% result(10,1)        %nombre entre 0 et 9;


% results(:,1)


weight_IH=2.*(rand(numH(1),numI)-0.5);   % [w11 W12 W13; W21 W22 W23]

weight_HH=cell(nHL-1);

for i=1:nHL-1
    weight_HH{i}=2.*(rand(numH(i+1),numH(i))-0.5);
end

weight_HO=2.*(rand(numO,numH(length(numH)))-0.5);%  [w11 12]

bias_H=cell(nHL-1);

for i=1:nHL
    bias_H{i}=2.*(rand(numH(i),1)-0.5); 
end

bias_O=2.*(rand(numO,1)-0.5);

lr=0.015;

res=0.1;


%show1Im(loadMNISTImages('ImMNIST1.png'))

        
%show1Im(input(:,1))



    

epoch=10;
trainingInputs=input(:,1:50000);
trainingResults=results(:,1:50000);
testingInputs=input(:,50001:60000);
testingResults=result(50001:60000);


for u=1:epoch
    randomIndex=randperm(size(trainingInputs,2));
for j=1:size(trainingInputs,2)
    
    train(trainingInputs(:,randomIndex(j)),trainingResults(:,randomIndex(j)))
%     if j==1
%         show1Im(trainingInputs(:,randomIndex(j)))
%         trainingResults(:,randomIndex(j))
%     end
    %show1Im(input(:,j))
    %draw(j)
%     if mod(j,size(input,2)/100)==0
%         disp(['Training: ',num2str(j/size(input,2)*100),'%'])
%     end
end


count=0;
tot=0;
for j=1:size(testingInputs,2)
    [M,I]=max(feedForward(testingInputs(:,j)));
    tot=tot+1;
%     if j==1
%         show1Im(testingInputs(:,j))
%         testingResults(:,j)
%     end
    if I-1==testingResults(j)
        count=count+1;
    end
    
end


SuccessRate=count/tot;

disp(['Epoch n° ', num2str(u),' => Success: ', num2str(SuccessRate*100),'%'])

end

% for j=1:10
%    feedForward(input(:,j)) 
% end

%show1Im(backProp([0.001; 0.981;  0.001; 0.001; 0.001; 0.001; 0.001; 0.001; 0.001; 0.001]))




%show1Im(photo2im28())
    function[O,H]=feedForward(input)
        H=cell(nHL);
        H{1}=weight_IH*input+bias_H{1};
        H{1}=1./(1+exp(-H{1}));
         
        for m=2:nHL
            H{m}=weight_HH{m-1}*H{m-1}+bias_H{m};
            H{m}=1./(1+exp(-H{m}));
        end
        
        O=weight_HO*H{nHL}+bias_O;
        O=1./(1+exp(-O));
        
    end  


    function[]=train(input,target)
        
        [O,H]=feedForward(input);
        outputError=target-O;
        
        O=O.*(1-O);
        dweight_HO=lr.*diag(outputError)*O*H{nHL}.';
        weight_HO=weight_HO+dweight_HO;
        bias_O=bias_O+lr.*diag(outputError)*O;
        
        hiddenLayerError=weight_HO.'*outputError;
        for n=nHL:-1:2
            H{n}=H{n}.*(1-H{n});
            dweight_HH=lr.*diag(hiddenLayerError)*H{n}*H{n-1}.';
            weight_HH{n-1}=weight_HH{n-1}+dweight_HH;
            bias_H{n}=bias_H{n}+lr.*diag(hiddenLayerError)*H{n};
            hiddenLayerError=weight_HH{n-1}.'*hiddenLayerError;
        end
        
        H{1}=H{1}.*(1-H{1});
        dweight_IH=lr.*diag(hiddenLayerError)*H{1}*input.';
        weight_IH=weight_IH+dweight_IH;
        bias_H{1}=bias_H{1}+lr.*diag(hiddenLayerError)*H{1};
        
        
    end


    function [Im] =backProp(results)
         H{nHL}=weight_HO\(sigInv(results)-bias_O)
         
        for n=nHL-1:-1:1
            disp('ok')
            H{n}=weight_HH{n}.'*((H{n+1})-bias_H{n+1});
        end
            sigInv(H{2})
            H{1}
            Im=weight_IH.'*((H{1})-bias_H{1});
            %Im=(Im);
            Im=1/(exp(-Im)+1);
            
    end

    function [x]=sigInv(x)
        x=-log(1./(x)-1);
        x=abs(x);
    end
    

     function[]= show1Im(input)
       scale=10;
       im=zeros(28*scale,28*scale);
        for o=1:28
            for p=1:28
                pix=input(p+(o-1)*28);
                im(scale*(p-1)+1:scale*p,scale*(o-1)+1:scale*o)=pix;
            end
        end
        imwrite(im, 'ImMSIST1.png')
        imshow 'ImMSIST1.png'
        
    end
   
    function[]=draw(j)
        if mod(j,100)==0
            pause(0.01)
            clf
            for k=0:res:1-res
                for l=0:res:1-res
                     pixColor=feedForward([k; l]);
                     rectangle('Position',[k l res res],'FaceColor',[pixColor pixColor pixColor]);
                end
            end
            
        end
    end

end

