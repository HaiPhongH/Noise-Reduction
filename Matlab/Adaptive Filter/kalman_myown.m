[signal, fs] = audioread('Data/strong_breath/strong9.wav');
R = 0.02;
L = length(signal);
noise = sqrt(R)*randn(L,1);
lp = dsp.FIRFilter('Numerator',fir1(31,[0.5 0.75]));
fnoise = lp(noise);

kalman = dsp.KalmanFilter('ProcessNoiseCovariance',0.00099,...
    'MeasurementNoiseCovariance',R,...
    'InitialStateEstimate', 5,...
    'InitialErrorCovarianceEstimate',1,...
    'ControlInputPort', false);

d = signal + fnoise;
e = kalman(d);

subplot(2,1,1);plot(0:L-1, signal(1:L), 0:L-1, e(1:L));
subplot(2,1,2);plot(0:L-1, e(1:L), 0:L-1, signal(1:L));
% 
%audiowrite('Filtered Data/kalman/strong_breath/strong10.wav',e,fs);
%audiowrite('Filtered Data/test_kal.wav',e,fs);

