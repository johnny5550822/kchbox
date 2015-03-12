% Convert the time-series into frequency-related features using Fast fourier transform (FFT)

function [new_s] = convertToFreqFeatures(t_s)
    for i=1:size(t_s,1)
        L = 100;    % Length of signal
        NFFT = 2^nextpow2(L); % Next power of 2 from length of y
        Fs = 100; %sampling frequency
        
        %FFT
        new_s = fft(t_s(i,:),NFFT);
        f = Fs/2*linspace(0,1,NFFT/2+1);
        y = 2*abs(new_s(1:NFFT/2+1))
        
        figure;
        plot(f,y) ;
        
    end


end












