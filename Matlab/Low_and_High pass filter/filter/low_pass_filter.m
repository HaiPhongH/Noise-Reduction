filename = 'E:\For Education\DO AN\My dataset\Breath\breath  (61).wav';
[y, Fs] = audioread(filename);
ax1 = subplot(2,1,1);
y = y(:,1);
dt = 1/Fs;
t = 0:dt:(length(y)*dt)-dt;
plot(ax1,t,y); xlabel('Seconds - Original'); ylabel('Amplitude')
ax2 = subplot(2,1,2);
Fn = Fs/2;
Fco = 3000;
Fsb = 500;
Rp = 1;
Rs = 10;
[n, Wn] = buttord(Fco/Fn, Fsb/Fn, Rp, Rs);
[b, a] = butter(n, Wn);
[sos, g] = tf2sos(b,a);
data_lpf = filtfilt(b, a, y);
audiowrite('E:\For Education\DO AN\My dataset\Breath\breath (61-low).wav', data_lpf, Fs);
plot(ax2, t,data_lpf);xlabel('Seconds - LPF'); ylabel('Amplitude');

