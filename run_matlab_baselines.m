clc ; clear ; close all

dataset_ids = [10, 14, 17, 45, 46, 52, 63, 109, 111, 145, 151];

for data_id=dataset_ids
    savefolder = sprintf('baseline_features/%d/dummy/ls/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/spec/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/udfs/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/ndfs/',data_id);
    mkdir(savefolder)

    % savefolder = sprintf('baseline_features/%d/dummy/rne/',data_id);
    % mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/dgufs/',data_id);
    mkdir(savefolder)

    % savefolder = sprintf('baseline_features/%d/dummy/inf_fs/',data_id);
    % mkdir(savefolder)
    % 
    % savefolder = sprintf('baseline_features/%d/dummy/fruar/',data_id);
    % mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/fmiufs/',data_id);
    mkdir(savefolder)

    savefolder = sprintf('baseline_features/%d/dummy/cnafs/',data_id);
    mkdir(savefolder)
end

for data_id=dataset_ids
    loadpath = sprintf('datasets/%d/dummy_pp_data.mat',data_id);
    savefolder = sprintf('baseline_features/%d/dummy/',data_id);
    X = load(loadpath).data;
    num_feat = size(X, 2);
    fprintf('dataset id: %d\n', data_id)
    for p=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fprintf('P: %.2f', p)
        k = round(p * num_feat);

        savepath = join([savefolder, sprintf('ls/features_%d.mat', 100*p)]);
        start = tic;
        X_red = ls_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")


        savepath = join([savefolder, sprintf('spec/features_%d.mat', 100*p)]);
        start = tic;
        X_red = spec_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")

        savepath = join([savefolder, sprintf('udfs/features_%d.mat', 100*p)]);
        start = tic;
        X_red = udfs_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")

        savepath = join([savefolder, sprintf('ndfs/features_%d.mat', 100*p)]);
        start = tic;
        X_red = ndfs_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")


        savepath = join([savefolder, sprintf('dgufs/features_%d.mat', 100*p)]);
        start = tic;
        X_red = dgufs_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")

        % savepath = join([savefolder, sprintf('rne/features_%d.mat', 100*p)]);
        % start = tic;
        % X_red = rne_model(X, k);
        % duration = toc;
        % num_red_feat = size(X_red, 2);
        % feat_ratio = num_red_feat / num_feat;
        % save(savepath, "X_red", "duration", "num_feat", ...
        %     "num_red_feat","feat_ratio")

        savepath = join([savefolder, sprintf('cnafs/features_%d.mat', 100*p)]);
        start = tic;
        X_red = cnafs_model(X, k);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")

        % savepath = join([savefolder, sprintf('inf_fs/features_%d.mat', 100*p)]);
        % start = tic;
        % X_red = inf_fs_model(X, k);
        % duration = toc;
        % num_red_feat = size(X_red, 2);
        % feat_ratio = num_red_feat / num_feat;
        % save(savepath, "X_red", "duration", "num_feat", ...
        %     "num_red_feat","feat_ratio")
    end
   fprintf('\n')
   for lammda=[0.1, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 2.]
       fprintf('lammda: %.2f', lammda)
        savepath = join([savefolder, sprintf('fmiufs/features_%d.mat', round(100*lammda))]);
        disp(savepath)
        start = tic;
        X_red = fmiufs_model(X, lammda);
        duration = toc;
        num_red_feat = size(X_red, 2);
        feat_ratio = num_red_feat / num_feat;
        save(savepath, "X_red", "duration", "num_feat", ...
            "num_red_feat","feat_ratio")

        % savepath = join([savefolder, sprintf('fruar/features_%d.mat', round(100*lammda))]);
        %start = tic;
        %X_red = fruar_model(X, lammda);
        %duration = toc;
        %num_red_feat = size(X_red, 2);
        %feat_ratio = num_red_feat / num_feat;
        %save(savepath, "X_red", "duration", "num_feat", ...
        %    "num_red_feat","feat_ratio")
   end

end