filename = 'E:\For Education\DO AN\My dataset\Cough\cough  (143).wav';
[y, Fs] = audioread(filename);
ax1 = subplot(2,1,1);
y = y(:,1);
dt = 1/Fs;
t = 0:dt:(length(y)*dt)-dt;
plot(ax1,t,y); xlabel('Seconds - Original'); ylabel('Amplitude');

ax3 = subplot(2,1,2);
[z,p,k] = butter(4, 1000/(Fs/2), 'high');
[sos, g] = zp2sos(z,p,k);
b = filtfilt(sos,g,y);
audiowrite('E:\For Education\DO AN\My dataset\Cough\cough  (143-high).wav', b, Fs);
plot(ax3, t,b);xlabel('Seconds - HPF'); ylabel('Amplitude');

