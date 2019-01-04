
[signal,fs] = audioread('Data/strong_breath/strong9.wav');
L = length(signal);
noise = rand(L,1);

lp = dsp.FIRFilter('Numerator',fir1(31,[0.5 0.75]));
% 
% nfilt = fir1(11,[0.5 0.75]);
% fnoise = filter(nfilt,1,noise);
%
fnoise = lp(noise);
M      = 32;                 % B?c - Chi?u dài vector tr?ng s?
delta  = 0.1;                % Kh?i t?o giá tr? hi?p ph??ng sai
P0     = (1/delta)*eye(M,M); % Kh?i t?o cho ma tr?n P c?a hi?p ph??ng sai
lambda = 0.99;               % Kh?i t?o forgetting factor
% Kh?i t?o b? l?c RLS
rlsfilt = dsp.RLSFilter(M, 'ForgettingFactor',lambda,...
    'InitialInverseCovariance',P0);


d = signal + fnoise;

[y,e] = rlsfilt(noise,d);

plot(0:L-1, d(1:L), 0:L-1, e(1:L));
title('RLS Output')
% subplot(2,1,1); plot(signal);title('Signal');
% subplot(2,1,2);plot(d);title('Noise+Signal');
% audiowrite('Filtered Data/rls/strong_breath/strong9.wav',e,fs);