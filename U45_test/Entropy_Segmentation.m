function R = Entropy_Segmentation(EW,ncg)
EW= imread('C:\Users\vasu\Desktop\matcodes\57.jpg');

if(nargin < 2)
    ncg = 6;
end

if(isempty(EW))
    disp('No input data');
    R = []; return;
end

[nr_bins,nr_max_neurons] = size(EW);    
nr_neurons = ncg;
nr_states = 2^nr_neurons;

% nr_groups: number of groups ncg neurons that are modeled with max entropy
Cg = combnk(1:nr_max_neurons,nr_neurons); 
nr_groups = size(Cg,1);  

% limit the number of groups
nr_max_groups = 5000;   % equivalent of Comb(15,6) - number of groups of 6 
% for a population of 15 units
if(nr_groups > nr_max_groups)
    rng('shuffle'); 
    rind = randi(nr_groups,[nr_groups,1]);
    IX = rind(1:nr_max_groups); 
    C = Cg(IX,:); 
    nr_groups = size(C,1);
else
    C = Cg;
end

% parameters for maximum entropy modeling
alpha = 0.1;    
nr_max_iter = 50000; 
lc = 0.1;   % level of change in local fields and interactions in percents

% compute Pn,P1 and P2 for all groups
Pn = zeros(nr_groups,nr_states); 
P1 = zeros(nr_groups,nr_states); 
H1 = cell(nr_groups,1); J1 = cell(nr_groups,1);
P2 = zeros(nr_groups,nr_states);
H2 = cell(nr_groups,1); J2 = cell(nr_groups,1);
Prf = cell(nr_groups,1);
% parfor loop
parfor k = 1:nr_groups
    EWg = zeros(nr_bins,nr_neurons);
    for j = 1:nr_neurons
        EWg(:,j) = EW(:,C(k,j));    %#ok<PFBNS>
    end
    Pn(k,:) = get_experimental_probability(EWg,nr_bins,nr_states);
    [P1(k,:),H1{k},J1{k},ni1] = max_entropy_one_probability(EWg,nr_neurons,...
        nr_states,alpha,nr_max_iter,lc);
    [P2(k,:),H2{k},J2{k},ni2] = max_entropy_two_probability(EWg,nr_bins,...
            nr_neurons,nr_states,alpha,nr_max_iter,lc);
    Prf{k} = get_performance_indexes(P1(k,:),ni1,P2(k,:),ni2,Pn(k,:));
    fprintf('group %d\n',k);
end

% Select the groups for which S1 > S2 > SN
K = 0;
for i = 1:nr_groups
    if((Prf{i}.S1 > Prf{i}.S2) && (Prf{i}.S2 > Prf{i}.Sn))
        K = K + 1;
        Pn(K,:) = Pn(i,:); 
        P1(K,:) = P1(i,:); H1{K} = H1{i}; J1{K} = J1{i};
        P2(K,:) = P2(i,:); H2{K} = H2{i}; J2{K} = J2{i};
        Prf{K} = Prf{i}; 
    end
end
if(K == 0)
    disp('No groups with S1 > S2 > SN');
    R = []; return;
end

Pn = Pn(1:K,:);
P1 = P1(1:K,:); H1 = H1(1:K); J1 = J1(1:K);
P2 = P2(1:K,:); H2 = H2(1:K); J2 = J2(1:K); 
Prf = Prf(1:K); 
nr_groups = K;

% get mean performance
f1 = zeros(nr_groups,1);
f2 = zeros(nr_groups,1);
for k = 1:nr_groups
    f1(k) = Prf{k}.f1;
    f2(k) = Prf{k}.f2;
end
mf1 = mean(f1); mf2 = mean(f2);

ME1.P = P1; ME1.H = H1; ME1.J = J1;
ME2.P = P2; ME2.H = H2; ME2.J = J2;
Perf.Prf = Prf;
Perf.f1 = f1; Perf.mf1 = mf1;
Perf.f2 = f2; Perf.mf2 = mf2;

R.Pn = Pn; R.ME1 = ME1; R.ME2 = ME2; R.Perf = Perf;