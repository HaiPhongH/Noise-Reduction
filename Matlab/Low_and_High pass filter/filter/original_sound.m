filename = 'E:\For Education\DO AN\My dataset\Cough\cough  (143).wav';
[y, Fs] = audioread(filename);
ax1 = subplot(1,1,1);
y = y(:,1);
dt = 1/Fs;
t = 0:dt:(length(y)*dt)-dt;
plot(ax1,t,y); xlabel('Seconds - Original'); ylabel('Amplitude');