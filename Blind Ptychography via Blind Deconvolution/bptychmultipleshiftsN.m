%Seperates a vector y into it's convoluted parts

clear all

%% Establishing the variables

L = 64; %number of measurements
delta = ceil(log2(L)); %Support of mask
Tests = 10;
Iterations = 1000;

errorx = zeros(2*delta,Tests);
errorm = zeros(2*delta,Tests);
errory = zeros(2*delta-1,1); 
errorY = zeros((2*delta-1)^2,1);
errorxN = zeros(6,4);
errormN = zeros(6,4);
signalnoiseratio = zeros(6,1);

counter2 = 0;
for Ntrue = 2:2:8 %Size of unknown part of signal
counter2 = counter2 + 1;
Y1 = (1/2)*ones(1000,1); Y2 = (1/2)*ones(1000,1); Y3 = (1/2)*ones(1000,1);

counter = 0;
for SNR=20:10:80
counter = counter + 1;
signalnoiseratio(counter) = SNR;
for test=1:Tests

F = dftmtx(L); %Let F be the DFT L X L matrix
Ctrue = (1/sqrt(sqrt(L)))*(randn(L,Ntrue) + 1i*randn(L,Ntrue)); 

truem = [randn(delta,1) + 1i*randn(delta,1); zeros(L-delta,1)];
xtrue = randn(Ntrue,1) +1i*randn(Ntrue,1);
truex = Ctrue*xtrue;

%% Noise

Y = zeros(L,L);
for k=0:L-1
    Y(:,k+1) = abs( fft(truex.*circshift(truem,-k), L) ).^2;
end

% SNR = 100;
 noise = rand(L,L) + 1i*rand(L,L);
 noise = (norm(Y)/10^(SNR/10))*noise/norm(noise);
 SNRdb = 10*log10(norm(Y)/norm(noise));

%% Computing measurements, convolutions and hadamard products
Y = zeros(L,L); truef = zeros(L,L); trueg = zeros(L,L);
trueY = zeros(L,L);
for k=0:L-1
      Y(:,k+1) = abs( fft(truex.*circshift(truem,-k), L) ).^2 + noise(:,k+1);
    truef(:,k+1) = flip(truem).*circshift(conj(flip(truem)),-k);
    trueg(:,k+1) = truex .* circshift(conj(truex),k);
    trueY(:,k+1) = cconv(truef(:,k+1),trueg(:,k+1),L);
end

trueYhad = fft(truef,L) .* fft(trueg,L);
Ytrue = conj(flip(transpose(fft(Y,L))))/L; %Ytrue = trueY
Ytruehad = conj((ifft(flip(transpose(fft(Y,L)))))); %Ytruehad = trueYhad

%% Khatri-Rao product of the signal
xnewbar = conj(xtrue); xnew = zeros(Ntrue^2,1);
for i=1:L
    for j=1:Ntrue^2
        k = mod(j,Ntrue);
        if k==0
            k = Ntrue;
        else
        end
        xnew(j) = xtrue(ceil(j/Ntrue))*xnewbar(k);
    end
end

xestimate = zeros(L,2*delta); 
mestimate = zeros(L,2*delta); 

%% Transposed Khatri-Rao Product of C
for shft=1:2*delta-1
    
    if shft<=delta
       Ctruebar = circshift(conj(Ctrue),shft-1,1);
    else
       Ctruebar = circshift(conj(Ctrue),shft-2*delta,1);
    end
C = zeros(L,Ntrue^2);
for i=1:L
    for j=1:Ntrue^2
        k = mod(j,Ntrue);
        if k==0
            k = Ntrue;
        else
        end
        C(i,j) = Ctrue(i,ceil(j/Ntrue))*Ctruebar(i,k);
    end
end



%% Blind Deconvolution via Wigner Gradient Descent

N = Ntrue^2;
A = conj(F*C);

if shft<=delta
    K = delta-shft+1;
    B = (1/sqrt(L))*F(:,L-delta+1:L-shft+1);
    ftrue = truef(:,shft);
    gtrue = trueg(:,shft);
    ytrue = Ytrue(:,shft);
else
    K = shft-delta;
    B = (1/sqrt(L))*F(:,L-(shft-delta)+1:L);
    ftrue = truef(:,L-2*delta+1+shft);
    gtrue = trueg(:,L-2*delta+1+shft);
    ytrue = Ytrue(:,L-2*delta+1+shft);
end

y = (1/sqrt(L))*fft(ytrue,L);

Bstar = B';
Astar = A';

%Computes the adjoint operator A*(y)
[U,S,V] = svd(linearAstar(y,Bstar,A,L),'econ');
d = S(1,1); h0 = U(:,1); x0 = V(:,1);
%Finds the leading singular value, left and right singular vectors of A*(y)
%Finds the initial starting point for the gradient descent

%mu = (L^2)*sqrt((L*(norm(B*h0, inf)^2))/(norm(h0)^2)); %Computes the incoherence constant
mu = 6*sqrt(L/(K+N))/log(L);
z0 = zeros(K,1);
options = optimoptions('fmincon','Display','off');
z = fmincon(@(z)argmin(z,d,h0),z0,[],[],[],[],[],[],@(z)incohcons(z,B,L,d,mu),options);
%Solves the optimization problem

u0 = z; v0 = sqrt(d)*x0; %Completes the initialization
u = u0; v = v0;
%e = (1/sqrt(2*L))*(randn(L,1) + 1i*randn(L,1)); %Defines the noise constant
rho = d^2/100; %Defines the rho constant
CL = d*(N*log(L)+ (rho*L)/((d*mu)^2));
eta = N/CL; %Sets the stepsize constant
I = Iterations; %Y0 = zeros(I,1); Y1 = zeros(I,1);
for t=1:I
W = 0;
for k=1:L
    W = W + 2*max([(L*abs(B(k,:)*u)^2)/(8*d*mu^2) - 1,0])*Bstar(:,k)*B(k,:)*u;
end
nablafh = (linearAstar(linearA(u*v',B,Astar,L)-y,Bstar,A,L))*v;
nablafx = (linearAstar(linearA(u*v',B,Astar,L)-y,Bstar,A,L))'*u;
nablagh = (rho/(2*d))*(2*max([(norm(u)^2)/(2*d) - 1,0])*u + (L/(4*mu^2))*W);
nablagx = (rho/(2*d))*2*max([(norm(v)^2)/(2*d) - 1,0])*v;

%Finding each of the gradients
W = 0;
for k=1:L
    W = W + max([(L*abs(B(k,:)*u)^2)/(8*d*mu^2) - 1,0])^2;
end
eta = 1;
    while Ftilde(u - eta*(nablafh + nablagh),v - eta*(nablafx + nablagx),y,B,Astar,L,rho,d,W) > Ftilde(u,v,y,B,Astar,L,rho,d,W) - eta*norm([nablafh + nablagh; nablafx + nablagx])^2
        eta = (1/2)*eta;
    end
%Iterating u & v
u = u - eta*(nablafh + nablagh);
v = v - eta*(nablafx + nablagx);
% %Correcting for norms
% normu = norm(u); 
% u = u*norm(ftrue)/normu;
% v = v*normu/norm(ftrue);


if shft<=delta
    f = [zeros(L - delta,1); u; zeros(shft-1,1)];
else
    f = [zeros(L - K,1); u];
end
g = C*conj(v);

Y1(t) = norm(cconv(f,g,L) - ytrue)^2/norm(ytrue)^2;
Y2(t) = norm(abs(f) - abs(ftrue))^2/norm(ftrue)^2;
Y3(t) = norm(abs(g) - abs(gtrue))^2/norm(gtrue)^2;
[Ntrue,SNR,test,shft,mod(t,1000),I/1000,eta,10*log10(Y3(t))]
end


%% Angular Synchronization

phaseOffset = angle( (f'*ftrue) / (ftrue'*ftrue) );
v = v * exp(-1i*phaseOffset); %Adjust for phase ambiguity
v = conj(v);
X = zeros(Ntrue,Ntrue);
for i=1:Ntrue
    for j=1:Ntrue
        X(i,j) = v(Ntrue*(i-1) + j);
    end
end

mags = sqrt( diag(X) );
[xrec, ~, ~] = eigs(X, 1, 'LM');    % compute leading eigenvector
xrec = xrec./abs(xrec);
xtrueest = sqrt(diag(X)) .* xrec;
truexest = Ctrue*xtrueest;
phaseOffset = angle( (truexest'*truex) / (truex'*truex) );
truexest = truexest* exp(1i*phaseOffset); %Adjust for phase ambiguity
errorxshift = 10*log10( norm(truexest - truex)^2/ norm(truex)^2 );

xestimate(:,shft) = truexest; 
errorx(shft,test) = errorxshift;
%% Computing Mask

truegest = zeros(L,L);
for k=0:L-1
    truegest(:,k+1) = truexest .* circshift(conj(truexest),k);
end
ffttruegest = fft(truegest,L);
ffttruefest = zeros(L,L);
for i=1:L
    for j=1:L
        ffttruefest(i,j) = Ytruehad(i,j)/ffttruegest(i,j);
    end
end
truefest = ifft(ffttruefest,L);
F1 = flip(truefest);
M = zeros(delta,delta);
for i=1:delta
    for j=1:i
        M(i,j) = F1(i,i-j+1);
    end
end
for i=1:delta
    for j=i+1:delta
        M(i,j) = conj(M(j,i));
    end
end
[mrec, ~, ~] = eigs(M, 1, 'LM');    % compute leading eigenvector
mrec = mrec./abs(mrec);
truemest = [sqrt( diag(M) ) .* mrec; zeros(L-delta,1)];
phaseOffset = angle( (truemest'*truem) / (truem'*truem) );
truemest = norm(truem)*(truemest* exp(1i*phaseOffset))/norm(truemest); %Adjust for phase ambiguity
errormshift = 10*log10(norm(truemest - truem)^2/norm(truem)^2);
mestimate(:,shft) = truemest; 
errorm(shft,test) = errormshift;
%%  Computing object error

truexest = truexest*alphaerror;
errorxshift = 10*log10( norm(truexest - truex)^2/ norm(truex)^2 );

xestimate(:,shft) = truexest; 
errorx(shft,test) = errorxshift;


%% Computing measurement errors
Yest=zeros(L,L);
for k=0:L-1
    Yest(:,k+1) = abs( fft(xestimate(:,shft).*circshift(mestimate(:,shft),-k), L) ).^2;
end
errory(shft,test) = norm(Yest - Y,'fro')^2/norm(Y,'fro')^2;
end

%[errorx errorm errory]

% [E,iy] = min(errory);

for i=1:2*delta
    for j=1:2*delta
        for k=0:L-1
    Yest(:,k+1) = abs( fft(xestimate(:,i).*circshift(mestimate(:,j),-k), L) ).^2;
        end
        errorY((i-1)*2*delta + j) = norm(Yest - Y,'fro')^2/norm(Y,'fro')^2;
    end
end
[E,iY] = min(errorY);
iYx = ceil(iY/(2*delta));
iYm = iY - (iYx-1)*2*delta;
truexest = xestimate(:,iYx);
truemest = mestimate(:,iYm);
%[errorx(iYx) errorm(iYm) errorY(iY)]

errorx(2*delta,test) = errorx(iYx,test);
errorm(2*delta,test) = errorm(iYm,test);


end
errorxN(counter,counter2) = mean(errorx(2*delta,:));
errormN(counter,counter2) = mean(errorm(2*delta,:));
end
end
%% Plotting figures

%First we plot our first figure, comparing the reconstruction error with
%the SNR for estimating object

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(signalnoiseratio, [errorxN(:,4) errorxN(:,3) errorxN(:,2) errorxN(:,1)],'LineWidth',1.5,'Parent',axes1);
set(plot1(1),'DisplayName','N = 8','Color',[0 0 0]);
set(plot1(2),'DisplayName','N = 6','LineStyle','-.','Color',[0 1 0]);
set(plot1(3),'DisplayName','N = 4','LineStyle',':','Color',[0 0 1]);
set(plot1(4),'DisplayName','N = 2','LineStyle','--','Color',[1 0 0]);

% Create ylabel
ylabel({'Reconstruction Error (in dB)'});

% Create xlabel
xlabel({'SNR (in dB)'});

% Create title
title({'SNR vs Reconstruction Error (Object)'});

box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'XTick',[20 30 40 50 60 70 80]);
% Create legend
legend(axes1,'show');

%Secondly, we plot our first figure, comparing the reconstruction error with
%the SNR for estimating mask

% Create figure
figure2 = figure;

% Create axes
axes2 = axes('Parent',figure2);
hold(axes2,'on');

% Create multiple lines using matrix input to plot
plot2 = plot(signalnoiseratio, [errormN(:,4) errormN(:,3) errormN(:,2) errormN(:,1)],'LineWidth',1.5,'Parent',axes2);
set(plot2(1),'DisplayName','N = 8','Color',[0 0 0]);
set(plot2(2),'DisplayName','N = 6','LineStyle','-.','Color',[0 1 0]);
set(plot2(3),'DisplayName','N = 4','LineStyle',':','Color',[0 0 1]);
set(plot2(4),'DisplayName','N = 2','LineStyle','--','Color',[1 0 0]);

% Create ylabel
ylabel({'Reconstruction Error (in dB)'});

% Create xlabel
xlabel({'SNR (in dB)'});

% Create title
title({'SNR vs Reconstruction Error (Mask)'});

box(axes2,'on');
hold(axes2,'off');
% Set the remaining axes properties
set(axes2,'XTick',[20 30 40 50 60 70 80]);
% Create legend
legend(axes2,'show');


%% Preassigned Functions
function[f] = argmin(z,d,h0)

f = norm(z - sqrt(d)*h0,2)^2;

end

function[f] = Ftilde(u,v,y,B,Astar,L,rho,d,W) 

f = norm(linearA(u*v',B,Astar,L) - y,2)^2 + rho*(max([(norm(u)^2)/(2*d) - 1,0])^2 + max([(norm(v)^2)/(2*d) - 1,0])^2 + W);

end

function[c,ceq] = incohcons(z,B,L,d,mu)

c = sqrt(L)*norm(B*z,inf)- 2*sqrt(d)*mu;
ceq = [];

end

function[f] = linearA(Z,B,Astar,L)

T = zeros(L,1);
for k=1:L
    T(k) = B(k,:)*Z*Astar(:,k);
end
f = T;

end

function[f] = linearAstar(y,Bstar,A,L)

Z = 0;
for k=1:L
    Z = Z + y(k)*Bstar(:,k)*A(k,:);
end
f = Z;

end
