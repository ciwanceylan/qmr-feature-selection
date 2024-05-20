clc ; clear ; close all

loadpath = 'datasets/isolet/Isolet1.mat';

savefolder = 'baseline_features/isolet/ls/';
mkdir(savefolder)

savefolder = 'baseline_features/isolet/spec/';
mkdir(savefolder)

savefolder = 'baseline_features/isolet/udfs/';
mkdir(savefolder)

savefolder = 'baseline_features/isolet/ndfs/';
mkdir(savefolder)

savefolder = 'baseline_features/isolet/fmiufs/';
mkdir(savefolder)

savefolder = 'baseline_features/isolet/cnafs/';
mkdir(savefolder)


savefolder = 'baseline_features/isolet/';
X = load(loadpath).X;
num_feat = size(X, 2);


disp('Running LS')
start = tic;
ls_I = ls_model(X);
ls_duration = toc(start);

disp('Running SPEC')
start = tic;
spec_I = spec_model(X);
spec_duration = toc(start);

disp('Running UDFS')
start = tic;
udfs_I = udfs_model(X);
udfs_duration = toc(start);

disp('Running NDFS')
start = tic;
ndfs_I = ndfs_model(X);
ndfs_duration = toc(start);

disp('Running CNAFS')
start = tic;
cnafs_I = cnafs_model(X);
cnafs_duration = toc(start);

disp('Running FMIUFS')
start = tic;
fmiufs_I = fmiufs_model(X);
fmiufs_duration = toc(start);


for k=[20, 30, 40, 50, 60, 70, 80, 90, 100]
    fprintf('K: %.2f', k)
    % 
    savepath = join([savefolder, sprintf('ls/features_%d.mat', k)]);
    X_red = X(:, ls_I(1:k));
    duration = ls_duration;
    save(savepath, "X_red", "duration")


    savepath = join([savefolder, sprintf('spec/features_%d.mat', k)]);
    X_red = X(:, spec_I(1:k));
    duration = spec_duration;
    save(savepath, "X_red", "duration")

    savepath = join([savefolder, sprintf('udfs/features_%d.mat', k)]);
    X_red = X(:, udfs_I(1:k));
    duration = udfs_duration;
    save(savepath, "X_red", "duration")

    savepath = join([savefolder, sprintf('ndfs/features_%d.mat', k)]);
    X_red = X(:, ndfs_I(1:k));
    duration = ndfs_duration;
    save(savepath, "X_red", "duration")

    savepath = join([savefolder, sprintf('cnafs/features_%d.mat', k)]);
    X_red = X(:, cnafs_I(1:k));
    duration = cnafs_duration;
    save(savepath, "X_red", "duration")

    savepath = join([savefolder, sprintf('fmiufs/features_%d.mat', k)]);
    X_red = X(:, fmiufs_I(1:k));
    duration = fmiufs_duration;
    save(savepath, "X_red", "duration")
end
fprintf('\n')

