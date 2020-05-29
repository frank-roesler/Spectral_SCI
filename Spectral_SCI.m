clear;

% Define interval:
a         = 30;
N         = 401;
ab        = linspace(-a, a, N);      % Domain in which basis functions e are constructed
n         = (N-1)/(max(ab)-min(ab));

a2        = 5;
l         = 201;
ab2       = linspace(-a2, a2, l);    % Domain of q
n2        = (l-1)/(max(ab2)-min(ab2));
ab2       = ab2(1:end-1);

C         = 50;

% Define potential q 
potential =@(x) 10i*x.*3.*exp(-x.^2);

x = ab2;
q = potential(x);

ab_shift = circshift(ab,-1);
ab_shift = ab_shift(1:end-1);
    
Laplace = n/3*diag(ab_shift.^3 - ab(1:end-1).^3);

e = zeros(N-1,l-1);
for k=1:N-1
    for j=1:length(ab2)
        if ab2(j)==0
            e(k,j) = -sqrt(n/2/pi)/n;
        else
            e(k,j) = -1i*sqrt(n/2/pi)*(exp(-1i*ab2(j).*ab(k+1)) - exp(-1i*ab2(j).*ab(k)))./ab2(j);
        end
    end
end

W_l = zeros(N-1);
for k=1:N-1
    for m=1:N-1
        W_l(k,m) = sum(q.*e(k,:).*conj(e(m,:)))/n2;
    end
end

K = Laplace + W_l;

%% Computation of Resonances:

M = 200;
X = linspace(-1,10, M);
Y = linspace(-10, 10, 2*M);
[XX,YY] = meshgrid(X,Y);
L = XX+1i*YY;

tic
parpool('local',2);

Resonances = zeros(size(L));
parfor k=1:size(L,1)*size(L,2)
    z = L(k);
    [R, flag]=chol((K-z*eye(N-1))'*(K-z*eye(N-1))-eye(N-1)/C^2);
    Resonances(k) = flag;
%     size(L,1)*size(L,2)-k
end
toc

Resonances = logical(Resonances);

figure;
%  subplot(2,1,2)
plot1 = plot(L(Resonances),'.','MarkerEdgeColor',[0.7 0.7 1], 'MarkerSize',20);
hold on
plot2 = plot(diag(Laplace)+0.0001i,'.','MarkerEdgeColor',[0.6,0.6,1], 'MarkerSize',20);
scatter1.MarkerEdgeAlpha = .2;
xlim([min(real(L(:))) max(real(L(:)))]);
ylim([min(imag(L(:))) max(imag(L(:)))]);
title('Spectrum:');

%% Comparison with Spectral Method:
LL=20;
NN=500;
xx=2*LL*(1-NN/2:NN/2)/NN;
column=[-NN^2/12-1/6 -.5*(-1).^(1:NN-1)./sin(pi*(1:NN-1)/NN).^2];
D2=(pi/LL)^2*toeplitz(column);
V = potential(xx);
A = -D2 + diag(V);
plot3 = plot(eig(A)+0.0i, 'x', 'MarkerSize',10,'MarkerEdgeColor',[0 0 0]);
legend([plot1;plot3],'Our Algorithm','Spectral Methods')
hold off

delete(gcp('nocreate'))







