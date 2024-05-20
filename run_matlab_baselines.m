clc ; clear ; close all

dataset_ids = [10, 14, 17, 45, 46, 52, 63, 109, 111, 145, 151];

for data_id=dataset_ids
    savefolder = sprintf('baseline_features/%d/factorize/ls/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/factorize/spec/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/factorize/udfs/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/factorize/ndfs/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/factorize/fmiufs/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/factorize/cnafs/',data_id);
    mkdir(savefolder)
end

for data_id=dataset_ids
    loadpath = sprintf('datasets/%d/factorized_pp_data.mat',data_id);
    savefolder = sprintf('baseline_features/%d/factorize/',data_id);
    X = load(loadpath).data;
    fprintf('Running dataset %d', data_id)
    
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

    num_feat = size(X, 2);
    fprintf('dataset id: %d\n', data_id)
    for p=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fprintf('P: %.2f', p)
        k = round(p * num_feat);

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

end