% Evalation matrices for the classifier performance

function [accuracy] = evaluation_matrice(true_label,pred_label)
    %parameters
    num_data = numel(true_label);
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    
    for i = 1:num_data
       if (true_label(i) == pred_label(i))
            if (true_label(i)==1)
                tp = tp + 1;
            else
                tn = tn + 1;
            end
       else
           if (true_label(i)==1)
               fn = fn + 1;
           else
               fp = fp + 1;
           end
       end        
    end
    
    %accuracy
    accuracy = (tp+tn)/num_data
end




