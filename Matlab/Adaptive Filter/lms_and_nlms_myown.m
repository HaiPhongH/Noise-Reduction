clear;
[signal,fs] = audioread('Data/strong_breath/strong9.wav');
L = length(signal);

%%
noise = rand(L,1);
nfilt = fir1(11,[0.5 0.75]); % B? l?c thông th?p
fnoise = filter(nfilt,1,noise);

%%
coeffs = nfilt.' -0.01;      % Kh?i t?o tr?ng s? c?a b? l?c
mu = 0.05;                   % Kích th??c b??c thích ?ng
d = signal + fnoise;
% Kh?i t?o b? l?c LMS v?i b?c là 12
lms = dsp.LMSFilter(12,'StepSize',mu,'InitialConditions',coeffs);
% Kh?i t?o l?c NLMS v?i b?c là 12
nlms = dsp.LMSFilter(12,'Method','Normalized LMS',...
    'StepSize',mu,'InitialConditions',coeffs);


[y_lms,e_lms,w_lms] = lms(noise,d);
[y_nlms,e_nlms,w_nlms] = nlms(noise,d);

%%
subplot(2,1,1); plot(0:L-1,signal(1:L),0:L-1,e_lms(1:L));title('LMS');
subplot(2,1,2); plot(0:L-1,signal(1:L),0:L-1,e_nlms(1:L));title('NLMS');
%%
% plot(0:L-1,signal(1:L),0:L-1,e(1:L));
% title('Noise Cancellation by the Sign-Data Algorithm');
% legend('Actual Signal','Result of Noise Cancellation',...
%        'Location','NorthEast');
% audiowrite('Filtered Data/lms/deep_breath/deep10.wav',e_lms,fs);
% audiowrite('Filtered Data/nlms/deep_breath/deep10.wav',e_nlms,fs);
