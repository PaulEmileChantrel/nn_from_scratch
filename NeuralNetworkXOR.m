function[]=NeuralNetworkXOR(numI,numH,numO)

numI=2;
numH=4;
numO=1;


global H;
global O;


weight_IH=2.*(rand(numH,numI)-0.5);   % [w11 W12 W13; W21 W22 W23]
weight_HO=2.*(rand(numO,numH)-0.5);   %  [w11 12]

bias_H=2.*(rand(numH,1)-0.5); 
bias_O=2.*(rand(numO,1)-0.5);

lr=0.1;

input=[1 1 0 0 ;1 0 1 0];
result=[0 1 1 0];


 squareSize=1;
res=0.05;
hold on



for j=1:10000
    
    i=randi(4);
    train(input(:,i),result(i))
    %draw(j)
end

test(input(:,1))
test(input(:,2))
test(input(:,3))
test(input(:,4))
% weight_IH
% weight_HO
% bias_O
% bias_H
%bias_H


    function[]=feedForward(input)
        H=weight_IH*input+bias_H;
        H=1./(1+exp(-H));

        O=weight_HO*H+bias_O;
        O=1./(1+exp(-O));
    end  


    function[]=train(input,target)
        
        feedForward(input)
        NNoutput=O;
        outputError=target-NNoutput;
        
        
        O=O.*(1-O);
        dweight_HO=lr.*diag(outputError)*O*H.';
        weight_HO=weight_HO+dweight_HO;
        bias_O=bias_O+lr.*diag(outputError)*O;
       
        hiddenLayerError=weight_HO.'*outputError;
        
        H=H.*(1-H);
        
        dweight_IH=lr.*diag(hiddenLayerError)*H*input.';
        weight_IH=weight_IH+dweight_IH;
        bias_H=bias_H+lr.*diag(hiddenLayerError)*H;
        
        
    end

    function[]=test(input)
        feedForward(input)
        O
    end

    
   
    function[]=draw(j)
        if mod(j,1000)==0
            pause(0.01)
            clf            
            for k=0:res:squareSize-res
                for l=0:res:squareSize-res
                     feedForward([k/squareSize; l/squareSize]);
                     pixColor=O;
                     rectangle('Position',[k l res res],'FaceColor',[pixColor pixColor pixColor]);
                end
            end
            
            
        end
    end

end

