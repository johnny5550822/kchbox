% Return evaluation matrices such as accuracy and precision from confusion
% matrix

function [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix] ... 
    =get_evaluation_matrice(confusion_matrix)
    
    tp = confusion_matrix(1,1);
    tn = confusion_matrix(2,2);
    fp = confusion_matrix(1,2);
    fn = confusion_matrix(2,1);
    
    acc = (tp+tn)/(tp+tn+fp+fn);
    sen = tp/(tp+fn);
    spec = tn/(tn+fp);
    pre = tp/(tp+fp);
    recall = tp/(tp+fn);
    f_measure = 2*(pre*recall)/(pre+recall);
    mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
end