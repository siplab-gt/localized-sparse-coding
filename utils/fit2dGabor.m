function results = fit2dGabor(data,options)
%FIT2DGABOR fit a gabor function to a two dimensional matrix (image)
%
% Uses the fit() function from the Curve Fitting Toolbox to fit a 2d Gabor
% function to data from a 2d matrix. fit() works with simple gradient descent.
% To avoid only finding a localy optimal solution this function samples over
% different starting values and selects the best result.
%
% A Gabor is a Gaussian in fourier space. So first a Gaussian is fit to the
% fourier transformed of the input data. Reasonable starting points are then
% extracted. Additional starting points are selected randomly.
%
% Extra fits are also perfomed to smoothed versions of the input data.
%
%
% - results
%   Structure containing the following fields:
%    patch     - evaluated fit of the same size as data
%    fit       - structure containig the coefficients of the fit function.
%                  Values are adjusted, namely the circular variables are
%                  bound reasonably. Also lambda is recalculated (see fit_raw).
%    fit_raw   - Matlab sfit result of the actual fit. Note that lambda depends
%                  on sigma here (multiple of full width at half maximum of
%                  gaussian envelope, in case of 'elliptical' in wave direction).
%    gof       - Matlab goodness of fit output (be careful, calculated on
%                   the smoothed data).
%    smoothness- sigma value for gaussian smoothing of the data.
%    r2        - coefficient of correlation. Note that it is calculated on
%                   the unsmoothed data (other than the gof values).
%    all_fits  - above described fields for all fits, sorted by r2. Only if
%                   option.getAllFits is set.
%
%
% - options
%   The following options are required:
%    shape     - either 'elliptical' or 'equal'
%                'equal' uses the  2d gabor filter function with
%                    xip =  (xi-x0)*cos(phi) + (yi-y0)*sin(phi);
%                    yip = -(xi-x0)*sin(phi) + (yi-y0)*cos(phi);
%                    G = b + a...
%                          * exp(-(xip .^2 + yip .^2)/2/sigma^2)...
%                         .* cos(2*pi*xip/lambda+phase);
%                'elliptical' extends the 2d gabor filter function with an
%                    elliptical gaussian basis function so that
%                    xip =  (xi-x0)*cos(phi) + (yi-y0)*sin(phi);
%                    yip = -(xi-x0)*sin(phi) + (yi-y0)*cos(phi);
%                    a_gauss =  cos(theta)^2/(2*sigmax^2) + sin(theta)^2/(2*sigmay^2);
%                    b_gauss = -sin(2*theta)/(4*sigmax^2) + sin(2*theta)/(4*sigmay^2);
%                    c_gauss =  sin(theta)^2/(2*sigmax^2) + cos(theta)^2/(2*sigmay^2);
%                    G = b + a
%                          * exp(-(a_gauss*xip.^2 - 2*b_gauss*xip.*yip + c_gauss* yip.^2))...
%                         .* cos(2*pi*xip/lambda+phase);
%                    Note that theta is relative to the wave direction.
%                    For elliptical gaussian functions see
%                    en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
%                custom function: add anonymous function to customGaborFit
%                                 add shape to makeGabor
%                                 add default to defaultBounds (might be omitted)
%                                 add custom case to getBetterSP
%                                 add permutation options to getstartpoints and randVecSP
%                                 add query for shape
%    runs      - number of different start-point-sets to be tested. Note:
%                minimum number is 48 (non-random starting points). Also,
%                there is always additional fits  for smoothed data in the end.
%                Reasonable value: 100
%
%   The following options will be set to default values if not specified:
%    parallel   - [true/false] uses Parallel Computing Toolbox | default = false
%    visualize  - [true/false] show current fit and best fit while running.
%                   | default = true
%    getAllFits - [true/false] save all runs in results.all_fits | default = true
%    ub,lb,sp   - vector of upper bounds, lower bounds, start points respectively
%                 for 'ellipse'
%                      [a, b, x0, y0, sigmax, sigmay, theta, phi, lambda,  phase]
%                 for 'equal'
%                      [a, b, x0, y0, sigma, phi, lambda, phase]
%                 Defaults are calculated from input.
%                 Note: lambda is a multiple of full width at half maximum (in
%                   wave direction in case of 'elliptical') of the gaussian
%                   envelope. Otherwise lambda can grow infinite in case of
%                   sinus-like gabor.
%                 Note: bounds for circular variables are set to -inf/inf to
%                   allow the fit to descent freely. Values are adjusted in
%                   the output.
%                 !!! make sp a matrix to give several different starting sets
%    sigmas     - vector of sigma values for gaussian smoothing grades (recommendation:
%                   ascending) | default = [0, 0.5, 1, 1.5, 2, 3]
%    r2         - set threshold for the coefficient of correlation at which
%                   fit is considered well enough. If N fits pass that
%                   threshold fitting is halted early. | default = 0.8
%    N          - number of fits that have to reach r2 to stop the fit
%                   early | default = 5
%
%
%
% Authors:
% Thede Witschel (thede.witschel(at)student.uni-tuebingen.de)
% Gerrit Ecke (gerrit.ecke(at)uni-tuebingen.de)
% Cognitive Neuroscience Unit
% Prof. Dr. H. A. Mallot
% Department of Biology
% University of Tuebingen


% this will start with a long list of options
% see if options were set
if nargin < 2
    options = struct();
end

% check wheter data is mxn matrix
if ~isnumeric(data) || ~ismatrix(data)
    error('Error: wrong data format. Input has to be a m times n matrix.')
end

% check whether options is a struct
if ~isstruct(options)
    error('Error: wrong options format. Please specify otions as struct.')
end

% ask for setting 'visualize'
if ~isfield(options,'visualize')
    fprintf('Visualization not set.\n Setting default value true.\n');
    options.visualize = true;
elseif ~(isscalar(options.visualize) && (islogical(options.visualize) || options.visualize==0 || options.visualize==1) )
    error('Detected options.visualize but is not boolean. Only boolean is supported.');
end
if options.visualize
    fig = gcf;
    clf; set(fig,'Name','Visualization of best and last fits','NumberTitle','off')
    subplot(1,3,1); imagesc(data); colormap gray; axis image; title({'Input data',[]});
    drawnow;
end

% ask for setting 'getAllFits'
if ~isfield(options,'getAllFits')
    fprintf('getRawFits not set. \n Setting default value false.\n')
    options.getAllFits=false;
elseif ~(isscalar(options.getAllFits) && (islogical(options.getAllFits) || options.getAllFits==0 || options.getAllFits==1) )
    error('Detected options.getAllFits but is not boolean. Only boolean is supported.');
end


% ask for shape and runs when they were not set
while (~isfield(options,'shape') || ~(strcmp(options.shape,'elliptical') || strcmp(options.shape,'equal')))
    fprintf(' No shape set or wrong string defined.\n Options are:\n elliptical\n equal\n');
    options.shape = input('please enter a string for the shape: ','s');
end

if ~isfield(options,'runs')
    fprintf('No number of start points set\n');
    options.runs = input('please enter how many start points should be started with: ');
elseif ~isnumeric(options.runs) || ~isreal(options.runs) || round(options.runs) ~= options.runs || options.runs < 1
    error('Detected options.runs but is not numeric or a natural number. Only natural numbers are supported.');
end

% start pool of parallel = true
if ~isfield(options,'parallel')
    options.parallel = false;
    disp('Not working in parallel')
    
elseif ~(isscalar(options.parallel) && (islogical(options.parallel) || options.parallel==0 || options.parallel==1) )
    error('Detected options.parallel but is not boolean. Only boolean is supported.');
end

switch options.parallel
    case true
        % global pool
        %if isempty(pool)
        pool=gcp();
        disp(['Using parallel pool. Number of workers: ' num2str(pool.NumWorkers)]);
        %end
end

% set N and r2 if none have been given
if ~isfield(options,'N')
    options.N = 5;
    fprintf(['No number of results past threshold r2 set\n default = ',num2str(options.N),'\n'])
elseif ~isnumeric(options.N) || ~isreal(options.r2) || round(options.N) ~= options.N || options.N < 1
    error('Detected options.N but is not numeric or a natural number. Only natural numbers are supported.');
end
if ~isfield(options,'r2')
    options.r2 = 0.8;
    fprintf(['No threshold r2 set\n default = ',num2str(options.r2),'\n'])
elseif ~isnumeric(options.r2) || ~isreal(options.r2) || options.r2 >= 1
    error('Detected options.r2 but is not numeric or higher then 1. Only real numbers lower or equal 1 are supported.');
end

% see if a vector of sigmas has been given and set defaults if not
if ~isfield(options,'sigmas')
    options.sigmas = [0, 0.5, 1, 1.5, 2, 3];
    fprintf(['No values for smoothing set\n default = ',mat2str(options.sigmas),'\n'])
elseif ~isvector(options.sigmas) || max(options.sigmas < 0)
    error('Detected options.sigmas but is not vector or negative. Only vectors containing positive values are supported.');
end

% see if boundaries have been given. Ipif not set defaults
if ~isfield(options,'lb') && ~isfield(options,'ub')
    defaultBounds = getdefaultBounds(options.shape, data);
    options.lb = defaultBounds.lb;
    options.ub = defaultBounds.ub;
    fprintf(['Using default values for upper and lower boundaries set by defaultBounds\n lower boundary: ',mat2str(defaultBounds.lb,2),'\n upper boundary: ',mat2str(defaultBounds.ub,2),'\n'])
elseif ~isfield(options,'lb') && isfield(options,'ub')==1
    defaultBounds = getdefaultBounds(options.shape, data);
    options.lb = defaultBounds.lb;
    fprintf(['Using default values for lower boundaries set by defaultBounds\n lower boundary: ',mat2str(defaultBounds.lb,2),'\n'])
    
elseif isfield(options,'lb')==1 && ~isfield(options,'ub')
    defaultBounds = getdefaultBounds(options.shape, data);
    options.ub = defaultBounds.ub;
    fprintf(['Using default values for upper boundaries set by defaultBounds\n upper boundary: ',mat2str(defaultBounds.ub,2),'\n'])
    
end
if ~isvector(options.lb) || max(~isnumeric(options.lb))
    error('Detected options.lb but is not vector with numeric values. Only vectors containing numeric values are supported.');
end
if ~isvector(options.ub) || max(~isnumeric(options.ub))
    error('Detected options.ub but is not vector with numeric values. Only vectors containing numeric values are supported.');
end

% see if start points have been given and set them in uniform dat structure
if ~isfield(options,'sp')
    if exist('defaultBounds')
        options.sp = defaultBounds.sp;
    else
        defaultBounds = getdefaultBounds(options.shape, data);
        options.sp = defaultBounds.sp;
    end
    fprintf(['Using default values for start points set by defaultBounds\n start points: ',mat2str(defaultBounds.sp,2),'\n'])
end
if ~isvector(options.sp) || max(~isnumeric(options.sp))
    error('Detected options.sp but is not vector with numeric values. Only vectors containing numeric values are supported.');
end

% do a prefit in fourier domain and add acquired parameters to start point permutation
[x,y] = meshgrid(1:size(data,2),1:size(data,1));
pf = fitGaussFF(x, y, data);
data = smoothPatches(data,options.sigmas);
patch = struct('data',data,'x',x,'y',y);
ptstats = struct('mean', mean(reshape(data{1},[],1)),'var',var(reshape(data{1},[],1)),'size',size(data{1}),'gauss',length(options.sigmas));
starts = getstartpoints(options.sp, pf, ptstats, options.runs);

results = struct('patch',[],'fit',[],'fit_raw',[],'gof',[],'smoothness',[],'r2',[]);
passed = 0;
bestR2SoFar=-9999;

switch options.parallel
    case true % here begins the parallised process
        % here begins the asynchronous parellel loop using parfeval
        for i_sp = 1:size(starts,1)
            para_res(i_sp) = parfeval(@customGaborFit,1,patch,starts(i_sp,:),options.lb,options.ub,options.shape);
        end
        cancelFutures = onCleanup(@() cancel(para_res));
end

for i_sp = 1:size(starts,1)
    
    switch options.parallel
        case true % here begins the parallised process
            %             cancelFutures = onCleanup(@() cancel(para_res));
            % here the futures acquired using parfeval are processed
            [fetch_idx,out] = fetchNext(para_res);
        case false
            out = customGaborFit(patch,starts(i_sp,:),options.lb,options.ub,options.shape);
            fetch_idx=i_sp;
    end
    
    results(i_sp).fit_raw = out.fit_raw;
    results(i_sp).fit = out.fit_adjusted;
    results(i_sp).gof = out.gof;
    results(i_sp).smoothness = options.sigmas(out.smooth);
    results(i_sp).r2 = out.r2;
    results(i_sp).patch = out.fit_evaluated;
    
    % count how many pass the threshold and stop if N has been reached
    if results(i_sp).r2 >= options.r2
        passed = passed + 1;
    end
    
    msg = ['Fetching ',int2str(fetch_idx),', ',int2str(passed),' above threshold'];
    disp(msg)
    
    if options.visualize
        set(0,'CurrentFigure',fig)
        % figure(fig);
        subplot(1,3,3); imagesc(out.fit_evaluated);
        colormap gray; axis image; title({'Last fit',['r2 = ',num2str(out.r2,2)]});
        if bestR2SoFar<out.r2
            subplot(1,3,2); imagesc(out.fit_evaluated);
            colormap gray; axis image; title({'Best fit so far',['r2 = ',num2str(out.r2,2)]});
            bestR2SoFar = out.r2;
        end
        drawnow;
    end
    
    if passed >= options.N
        disp('Threshold reached, exiting the loop')
        switch options.parallel
            case true
                cancel(para_res);
        end
        break
    end
    
end


% check if any results came out empty
temp_ind=true(size(results,2),1);
for i_empty = 1:length(results)
    if isempty(results(i_empty).r2);
        temp_ind(i_empty) = false;
    end
end
results = results(temp_ind);

% use parameters of the  best fits to attempt getting even better fits
disp('Attempt to get better results using best fits of smoothed versions as starting points...')

starts = getBetterSP(results,starts,options);


% On Linux calling parfeval directly after cancelling leads to
% crashing parallel pools (Matlab 2016b). This is how we act on this.
for i = 1 : 4
    try
        switch options.parallel
            case true
                
                for i_refit = 1:size(options.sigmas(options.sigmas~=0),2)
                    rf_res(i_refit) = parfeval(@customGaborFit,1,patch,starts(end-size(options.sigmas(options.sigmas~=0),2)+i_refit,:),options.lb,options.ub,options.shape);
                end
                cancelFutures = onCleanup(@() cancel(para_res));
        end
        
        
        res_l = length(results);
        for i_rf = 1 : size(options.sigmas(options.sigmas~=0),2)
            switch options.parallel
                case true
                    %             cancelFutures = onCleanup(@() cancel(para_res));
                    %try
                    [fetch_idx,out] = fetchNext(rf_res);
                    %catch
                    %    keyboard
                    %end
                case false
                    out = customGaborFit(patch,starts(end-size(options.sigmas(options.sigmas~=0),2)+i_rf,:),options.lb,options.ub,options.shape);
                    fetch_idx=i_rf;
            end
            results(res_l + i_rf).fit_raw = out.fit_raw;
            results(res_l + i_rf).fit = out.fit_adjusted;
            results(res_l + i_rf).gof = out.gof;
            results(res_l + i_rf).smoothness = options.sigmas(out.smooth);
            results(res_l + i_rf).r2 = out.r2;
            results(res_l + i_rf).patch = out.fit_evaluated;
            
            if results(res_l + i_rf).r2 >= options.r2
                passed = passed + 1;
            end
            
            msg = ['Fetching ',int2str(res_l + fetch_idx),', ',int2str(passed),'p above threshold'];
            disp(msg)
            
            if options.visualize
                set(0,'CurrentFigure',fig)
                % figure(fig);
                subplot(1,3,3); imagesc(out.fit_evaluated);
                colormap gray; axis image; title({'Last fit',['r2 = ',num2str(out.r2,2)]});
                if bestR2SoFar<out.r2
                    subplot(1,3,2); imagesc(out.fit_evaluated);
                    colormap gray; axis image; title({'Best fit so far',['r2 = ',num2str(out.r2,2)]});
                    bestR2SoFar = out.r2;
                end
                drawnow;
            end
        end
        break;
        
    catch
        %system(['touch parallelPoolCrash_',datestr(now,'yyyy_mm_dd_HH_MM_SS')])
        if i == 4
            error('Failed 3 attempts to reopen parallel pool. Exiting.');
        else
            disp('Parallel pool failed. Trying to reopen...');
        end
    end
end

% check if any results came out empty
temp_ind=true(size(results,2),1);
for i_empty = 1:length(results)
    if isempty(results(i_empty).r2);
        temp_ind(i_empty) = false;
    end
end
results = results(temp_ind);

% sort r2 of results
Hr2 = cat(1,results.r2);
[~,sr2] = sort(Hr2);
results_temp = results(sr2);

clear results;
results=results_temp(end);
if options.getAllFits
    results.allFits=results_temp;
end
end


function coll = fitGaussFF(x,y,z)
%FITGAUSSFF is called by fit2dGabor as a prefit to get additional
% reasonable starting values for the main fit
%--------------------------------------------
%x, y - meshgrid of data size
% z - pixel values of data

patchsize = size(z);
% fit gaussian in fourier domain
gauss2fun = @(a, b, x0, y0, sigma,x,y)(a*exp(-((x-x0).^2+(y-y0).^2)/2/sigma^2)+b);
%Start with fitting a Gaussian in the Fourier domain
zif = log(fftshift(abs(fft2(z))) + 1);

% highly rudimentary high-pass filter
[rr, cc] = meshgrid(1:patchsize(2),1:patchsize(1));
C = sqrt((rr-patchsize(2)/2).^2+(cc-patchsize(1)/2).^2)>=2;
zifC = log(1+(zif .* C));

%smooth a little, this helps
g = [.2,1,.2];
g = g'*g;

zifC = conv2(zifC-mean(zifC(:)),g,'same');

ss = @(x) x(1:end-1);

%Because of wraparound, do it in two shots
[xip,yip] = meshgrid((linspace(-.5,0,size(x,2)/2)),ss(linspace(-.5,.5,size(x,1)+1)));
zifA = zifC(:,1:size(xip,2));
zifR = reshape(zifA,[numel(zifA),1]);
xipR = reshape(xip,[numel(xip),1]);
yipR = reshape(yip,[numel(yip),1]);
[r1, gof_1] = fit([xipR,yipR],zifR,gauss2fun,...
    'StartPoint',[max(zifR)-min(zifR), min(zifR), -0.25, 0, 0.25] ,...
    'MaxIter',10^4,...
    'MaxFunEvals',10^4);


[xip,yip] = meshgrid(ss(linspace(-.5,.5,size(x,2)+1)),linspace(-.5,0,size(x,1)/2));
zifB = zifC(1:size(xip,1),:);
zifR = reshape(zifB,[numel(zifB),1]);
xipR = reshape(xip,[numel(xip),1]);
yipR = reshape(yip,[numel(yip),1]);
[r2, gof_2] = fit([xipR,yipR],zifR,gauss2fun,...
    'StartPoint',[max(zifR)-min(zifR), min(zifR), 0, -0.25, 0.25] ,...
    'MaxIter',10^4,...
    'MaxFunEvals',10^4);

%Look at the quality of each fit to decide which one to use
if gof_1.sse < gof_2.sse
    %Go with r1
    coords = [r1.x0,r1.y0,r1.sigma];
else
    coords = [r2.x0,r2.y0,r2.sigma];
end

%0 degrees is vertical
phi = atan2(coords(2),coords(1));
while phi > pi || phi <= 0
    phi = phi - sign(phi) * pi;
end


sf = min(sqrt(coords(1).^2+coords(2).^2),.5);
dx = x(1,2)-x(1,1);
dy = y(2,1)-y(1,1);
da = .5*(dx+dy);
lambda = min(1/sf*da, 2*max(size(x))-1);
sigma  = da/coords(3)/pi/2/1.3;

coll = struct('sigma',sigma,'phi',phi,'lambda',lambda);
end

function r2 = getrsquare(data1,data2)
% GETRSQUARE is called by fit2dGabor to evaluate how good the final fit
% actually is
%-------------
% data1 - original data/ observed data
% data2 - data produced by model
% %
mean1 = mean(reshape(data1,[],1));

ss_tot = sum((reshape(data1,[],1)-mean1).^2);

ss_res = sum((reshape(data1,[],1)-reshape(data2,[],1)).^2);

r2 = 1 - ss_res/ss_tot;

end
function anarray = smoothPatches(patch, sigma)
%SMOOTHPATCHES is called by fit2dGabor to get smoothed versions of
%the data
%----------
% patch - matrix of image/data
% sigma - vector of sigmas of the gaussian kernel
anarray = cell(length(sigma),1);
for i_smooth = 1:length(sigma)
    if sigma(i_smooth) ~= 0
        g = fspecial('gaussian', ceil(sigma(i_smooth)*3), sigma(i_smooth));
        anarray{i_smooth} = conv2(patch-mean(patch(:)),g,'same');
    else
        anarray{i_smooth} = patch;
    end
end
end
function out = getstartpoints(sp, pf, stats, runs)
% GETSTARTPOINTS is called by fit2dGabor to get a set of starting values for the fitting
% process based on given start points (sp), combinations of start points
% and prefit results (pf) and random values between the lower and upper
% boundaries (lb, ub)
% stats relate to image mean and variance
% past_out and base_in are optional, if given the prefit and sp combination
% will not be calculated (base) and the vector past_out will be extended
% instead of creating a new out
% ------------------------------------------
% only works with vector lengths of 8 or 10
%
% sp - starting points given to the function
% pf - parameters acquired from prefit
% stats - properties of the data
% runs - base number of values to be acquired (end result will be
% higher)
if length(sp) == 10
    max_sigma = max([sp(4),sp(5)]);
    sp_short = [max_sigma,sp(8),sp(9)];
elseif length(sp) == 8
    sp_short = ([4,6,7]);
else
    error('getstartpoints only defined for vectors of length 8 and 10')
end

% first one is the given starting point, not needed as that is the first
% permutation
%out(1,:) = sp;
out = [];

% after that take all permutations of start sp and the three
% prefit results
gauss = stats.gauss;
pf = [pf.sigma, pf.phi, pf.lambda];
combsOfPF = allperms([0,1],length(pf));

for i_combs = 0:size(combsOfPF,1) * gauss - 1
    tmp_pf = combsOfPF(1 + floor(i_combs/gauss),:).*pf + (1-combsOfPF(1 + floor(i_combs/gauss),:)).*sp_short;
    tmp_sp = sp;
    if length(sp)==10
        tmp_sp([5,6]) = tmp_pf(1); % sigma is set to prefit value
        tmp_sp([8,9]) = tmp_pf([2,3]);
    else
        tmp_sp([5,6,7]) = tmp_pf;
    end
    out = [out; tmp_sp, mod(i_combs, stats.gauss)];
    fix_sp = i_combs + 1;
end

past_out = size(out,1);

% here we randomly combine the previous results with values generated
% according to randVecSP, we always create n new start points per
% run
for i_rand = size(out,1)+1 : runs
    base_sp = out(randi(fix_sp,1),:);
    bb_vec = randVecSP(stats,length(sp));
    
    tmp_gauss = base_sp(end);
    tmp_base = base_sp(1:end-1);
    tmp_vec = randi([0,1], size(tmp_base));
    out(i_rand,:) = [tmp_vec.*bb_vec + (1-tmp_vec).* tmp_base, tmp_gauss];
    
end
out=unique(out,'rows','stable');
end
function combs = allperms(values, k)
%ALLPERMS is called by getstartpoints to get all permutations of two values
set = cell(1,k);
for i=1:k
    set{i} = values;
end
n=numel(set);
combs=cell(n,1);
[combs{1:n}]=ndgrid(set{end:-1:1});
combs=reshape(cat(n+1,combs{:}),[],n);
combs=combs(:,end:-1:1);
end
function Vec = randVecSP(stats, n)
%RANDVECSP is called by getstartpoints
% this function sets the boundaries in which random starting values
% will be generated
% ------------------
% only works with parameter vectors of size 10 or 8
if n == 10
    Vec = zeros(1,10);
    Vec(1) = 10 * stats.var * (rand(1)*2-1);
    Vec(2) = stats.mean + 3 * stats.var * (rand(1)*2-1);
    Vec(3) = stats.size(1)/2 + (stats.size(1)/2-1) * (rand(1)*2-1);
    Vec(4) = stats.size(2)/2 + (stats.size(2)/2-1) * (rand(1)*2-1);
    Vec([5,6]) = [stats.size(1)/2 * rand(1), stats.size(2)/2 * rand(1)];
    Vec(7) = pi/4 *(rand(1)*2-1);
    Vec(8) = pi/2 *(rand(1)*2-1);
    Vec(9) = rand(1)*1.95+0.05;%(mean([stats.size(1),stats.size(2)])-2) * rand(1) + 2;
    Vec(10)= pi *(rand(1)*2-1);
else
    Vec = zeros(1,8);
    mean_size = mean([stats.size(1),stats.size(2)]);
    Vec(1) = 10 * stats.var * (rand(1)*2-1);
    Vec(2) = stats.mean + 3 * stats.var * (rand(1)*2-1);
    Vec(3) = stats.size(1)/2 + (stats.size(1)/2-1) * (rand(1)*2-1);
    Vec(4) = stats.size(2)/2 + (stats.size(2)/2-1) * (rand(1)*2-1);
    Vec(5) = mean_size/2 * rand(1);
    Vec(6) = pi/2 *(rand(1)*2-1);
    Vec(7) = rand(1)*1.95+0.05;%(mean_size-2) * rand(1) + 2;
    Vec(8) = pi * (rand(1)*2-1);
    
end
end
function defaultBounds = getdefaultBounds(shape, data)
%DEFAULTBOUNDS is called by fit2dGabor to get starting points and
% boundaries for the fit if they are not provided
%--------------------------------------------------
%shape - version of Gabor function to be used
% datahalf - half the dimensions of original data
datasize=[size(data,2), size(data,1)];
defaultBounds = struct('lb',[],'ub',[],'sp',[]);
switch shape
    case 'elliptical'
        %                    [a,                b,             x0,            y0,            sigmax,        sigmay,        theta, phi, lambda,          phase]
        defaultBounds.lb   = [0,             -inf,             1,             1,             .5,            .5,           -inf,  -inf, 0.05,           -inf  ];
        % defaultBounds.lb = [0,             -inf,            -inf,          -inf,           0,             0,            -inf,  -inf, 0,              -inf  ];
        % defaultBounds.ub = [inf,            inf,             datasize(1)-1, datasize(2)-1, datasize(1)/2, datasize(2)/2, inf,   inf, max(datasize),   inf  ];
        defaultBounds.ub   = [inf,            inf,             datasize(1)-1, datasize(2)-1, datasize(1)/2, datasize(2)/2, inf,   inf, 2,               inf  ];
        % defaultBounds.ub = [inf,            inf,             inf,           inf,           inf,           inf,           inf,   inf, inf,             inf  ];
        % defaultBounds.sp = [max(data(:))/2, median(data(:)), datasize(1)/2, datasize(2)/2, datasize(1)/3, datasize(2)/3, 0,     0,   min(datasize)/2, 0    ];
        defaultBounds.sp   = [max(data(:))/2, median(data(:)), datasize(1)/2, datasize(2)/2, datasize(1)/3, datasize(2)/3, 0,     0,   1.5,             0    ];
        
    case 'equal'
        %                    [a,              b,               x0,            y0,            sigma,           phi, lambda,          phase]
        defaultBounds.lb   = [0,             -inf,             1,             1,             .5,             -inf, 0.05,           -inf  ];
        % defaultBounds.lb = [0,             -inf,            -inf,          -inf,           0,              -inf, 0,              -inf];
        % defaultBounds.ub = [inf,            inf,             datasize(1)-1, datasize(2)-1, max(datasize)/2, inf, max(datasize),   inf];
        defaultBounds.ub   = [inf,            inf,             datasize(1)-1, datasize(2)-1, max(datasize)/2, inf, 2,               inf];
        % defaultBounds.ub = [inf,            inf,             inf,           inf,           inf,             inf, inf,             inf];
        % defaultBounds.sp = [max(data(:))/2, median(data(:)), datasize(1)/2, datasize(2)/2, max(datasize)/3, 0,   min(datasize)/2, 0];
        defaultBounds.sp   = [max(data(:))/2, median(data(:)), datasize(1)/2, datasize(2)/2, max(datasize)/3, 0,   1.5            , 0];
        
end

end
function out = customGaborFit(data,sp,lb,ub,shape)
%ELLIPTICALGABORFIT is called by fit2dGabor
% here the actual fitting is happening using the matlab native
% function 'fit'
%-----------------
% data - 'observed' data, rather the image you want to be fitted
% sp - starting points for the matlab function 'fit'
% lb, ub - upper and lower boundaries for matlabs function 'fit'
% shape - either 'elliptical' or 'equal'

g_level = sp(end)+1;
sp = sp(1:end-1);
patchCGF = data(g_level).data;

patchCGF = reshape(patchCGF,[],1);
xCGF = reshape(data(1).x,[],1);
yCGF = reshape(data(1).y,[],1);
% here is the anonymous function that will be used
switch shape
    case 'elliptical'
        xip =  @(x0, y0, phi, x, y) (x-x0)*cos(phi) + (y-y0)*sin(phi);
        yip =  @(x0, y0, phi, x, y) -(x-x0)*sin(phi) + (y-y0)*cos(phi);
        Cov_a = @(sigmax, sigmay, theta)  cos(theta)^2/(2*sigmax^2) + sin(theta)^2/(2*sigmay^2);
        Cov_b = @(sigmax, sigmay, theta) -sin(2*theta)/(4*sigmax^2) + sin(2*theta)/(4*sigmay^2);
        Cov_c = @(sigmax, sigmay, theta)  sin(theta)^2/(2*sigmax^2) + cos(theta)^2/(2*sigmay^2);
        gabor2fun = @(a, b, x0, y0, sigmax, sigmay,theta, phi, lambda,  phase, x, y)...
            a*exp(...
            -( Cov_a(sigmax, sigmay, theta)*xip(x0, y0, phi, x, y).^2 ...
            -  2*Cov_b(sigmax, sigmay, theta)*xip(x0, y0, phi, x, y).*yip(x0, y0, phi, x, y) ...
            +  Cov_c(sigmax, sigmay, theta)*yip(x0, y0, phi, x, y).^2)) ...
            .* cos(2*pi*xip(x0, y0, phi, x, y)/(lambda*2.3548*(cos(theta)^2*sigmax+sin(theta)^2*sigmay))+phase) + b;
        
        % this is were the fit actually happens using the matlab
        % default gradient descent
        [params,gof] = fit([xCGF, yCGF], patchCGF,gabor2fun, ...
            'StartPoint', sp, ...
            'Lower', lb, ...
            'Upper', ub, ...
            'MaxIter',10^4,...
            'MaxFunEvals',10^4,...
            'TolFun', 10^-6,...
            'TolX', 10^-7);
        
    case 'equal'
        xip =  @(x0, y0, phi, x, y) (x-x0)*cos(phi) + (y-y0)*sin(phi);
        yip =  @(x0, y0, phi, x, y) -(x-x0)*sin(phi) + (y-y0)*cos(phi);
        
        gabor2fun = @(a,b, x0, y0, sigma, phi, lambda, phase, x, y) ...
            a* exp(-(xip(x0, y0, phi, x, y).^2+yip(x0, y0, phi, x, y).^2)/2/sigma^2)...
            .* cos(2*pi*xip(x0, y0, phi, x, y)/(lambda*2.3548*sigma)+phase) + b;
        
        % this is were the fit actually happens using the matlabnow
        % default gradient descent
        [params,gof] = fit([xCGF, yCGF], patchCGF,gabor2fun, ...
            'StartPoint', sp, ...
            'Lower', lb, ...
            'Upper', ub, ...
            'MaxIter',10^4,...
            'MaxFunEvals',10^4,...
            'TolFun', 10^-6,...
            'TolX', 10^-7);
end
params_adjusted = postprocessValues(params);
fit_evaluated = makeGabor(params_adjusted,xCGF,yCGF,shape);
r2 = getrsquare(data(1).data, fit_evaluated);
fit_evaluated=reshape(fit_evaluated,size(data(1).data));
out = struct('fit_raw',params,'fit_adjusted',params_adjusted,'fit_evaluated',fit_evaluated,'gof',gof,'smooth',g_level,'shape',shape,'r2',r2);

end
function G = makeGabor(means, xi, yi, shape)
% this function is used to produce 2D-matrices using the parameters
% acquired from the fit and the same function as used for the fit,
% but not written as anonymous function
%----------------------------------------
%means - the mean parameters acquired by fitting
%xi,yi - meshgrid of matrix size
%shape - version of Gabor function to be used
switch shape
    case 'elliptical'
        xip =  (xi-means.x0)*cos(means.phi) + (yi-means.y0)*sin(means.phi);
        yip = -(xi-means.x0)*sin(means.phi) + (yi-means.y0)*cos(means.phi);
        
        a_gauss =  cos(means.theta)^2/(2*means.sigmax^2) + sin(means.theta)^2/(2*means.sigmay^2);
        b_gauss = -sin(2*means.theta)/(4*means.sigmax^2) + sin(2*means.theta)/(4*means.sigmay^2);
        c_gauss =  sin(means.theta)^2/(2*means.sigmax^2) + cos(means.theta)^2/(2*means.sigmay^2);
        
        G = means.a * exp(-(a_gauss*xip.^2 - 2*b_gauss*xip.*yip + c_gauss*yip.^2))...
            .* cos(2*pi*xip/means.lambda+means.phase) + means.b;
        
    case 'equal'
        xip =  (xi-means.x0)*cos(means.phi) + (yi-means.y0)*sin(means.phi);
        yip = -(xi-means.x0)*sin(means.phi) + (yi-means.y0)*cos(means.phi);
        
        G = means.a * exp(-(xip .^2 + yip .^2)/2/means.sigma^2)...
            .* cos(2*pi*xip/means.lambda+means.phase) + means.b;
        
end
end
function startsImp = getBetterSP(results,startsImp,options)
%GETBETTERSP is called by fit2dGabor
% it selects the best fitting results and prepares them to be used
% for another round of fitting
%-----------------------------
% startsImp - matrix of start points used for the fitting
% results - fits acquired by using the start points found in
% startsImp
% options - options given to fit2dGabor, only shape is needed


% sort r2 of results
Hr2 = cat(1,results.gof);
Hr2 = cat(1,Hr2.rsquare);
sr2 = sort(Hr2);

% add parameters of best fits as start points
i_tmp_start=1;
desiredValues = options.sigmas(options.sigmas~=0);


while size(desiredValues,2) > 0 ;
    %for i_tmp_start = 0:3
    tmp = find(Hr2 == sr2(end - i_tmp_start+1));
    if min(results(tmp(1)).smoothness~=desiredValues)
        i_tmp_start=i_tmp_start+1;
        if i_tmp_start >= size(sr2,1);
            break;
        end
        continue;
    end
    tmp_res = results(tmp(1)).fit;
    switch options.shape
        case 'elliptical'
            tmp_start = [tmp_res.a,tmp_res.b,tmp_res.x0,tmp_res.y0,tmp_res.sigmax,tmp_res.sigmay,tmp_res.theta,tmp_res.phi,tmp_res.lambda,tmp_res.phase,0];
        case 'equal'
            tmp_start = [tmp_res.a,tmp_res.b,tmp_res.x0,tmp_res.y0,tmp_res.sigma,tmp_res.phi,tmp_res.lambda,tmp_res.phase,0];
    end
    sizeProbe=size(startsImp,1);
    startsImp = cat(1,startsImp,tmp_start);
    startsImp=unique(startsImp,'rows','stable');
    if size(startsImp,1) ~= sizeProbe
        desiredValues=desiredValues(~(results(tmp(1)).smoothness==desiredValues));
    end
    if i_tmp_start >= size(sr2,1);
        break;
    end
    i_tmp_start=i_tmp_start+1;
end
end
function params_adjusted = postprocessValues(params)

coeffn=coeffnames(params);
coeffv=coeffvalues(params);

params_adjusted = struct();
for i = 1 : size(coeffn,1)
    params_adjusted=setfield(params_adjusted,coeffn{i},coeffv(i));
end

% adjusting phi
while params_adjusted.phi > pi || params_adjusted.phi <= -pi
    params_adjusted.phi = params_adjusted.phi - sign(params_adjusted.phi) * 2 * pi;
end
% because of symmetry phi should be between -pi/2 and pi/2
if params_adjusted.phi > pi/2 || params_adjusted.phi <= -pi/2
    params_adjusted.phi = params_adjusted.phi - sign(params_adjusted.phi) * pi;
    params_adjusted.phase = -params_adjusted.phase; % phase has to be switched
    if isfield(params_adjusted,'theta')
        params_adjusted.theta = params_adjusted.theta + pi; % theta has to be turned back (is relative to phi)
    end
end

% adjusting theta
if isfield(params_adjusted,'theta')
    while params_adjusted.theta > pi || params_adjusted.theta <= -pi
        params_adjusted.theta = params_adjusted.theta - sign(params_adjusted.theta) * 2 * pi;
    end
    % because of symmetry theta should be between -pi/2 and pi/2
    if params_adjusted.theta > pi/2 || params_adjusted.theta <= -pi/2
        params_adjusted.theta = params_adjusted.theta - sign(params_adjusted.theta) * pi;
    end
end

while params_adjusted.phase > pi || params_adjusted.phase <= -pi
    params_adjusted.phase = params_adjusted.phase - sign(params_adjusted.phase) * 2 * pi;
end

if isfield(params_adjusted,'sigma')
    params_adjusted.lambda = params.lambda * 2.3548 * params.sigma;
elseif isfield(params_adjusted,'sigmax') && isfield(params_adjusted,'sigmay')
    params_adjusted.lambda = params.lambda * 2.3548 * (cos(params.theta)^2*params.sigmax+sin(params.theta)^2*params.sigmay);
end
end

