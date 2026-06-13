%% OpenBCI 8-Channel LSTM Classifier - CORRECTED (review findings 2, 3, 5, 6)
% Changes vs train_openbci_lstm_classifier.m:
%   - Split by SEGMENT (keyed on file path), never random window
%     (Finding 2: 0.25 s stride on 2 s windows = 87.5% overlap; a random
%      window split puts near-duplicate windows in both train and test)
%   - Leave-one-segment-out CV, predictions pooled across folds
%     (Finding 6: ~10 segments is the real sample count, not ~thousands
%      of overlapping windows)
%   - Logistic-regression baseline on 3 hand features
%     [mean pairwise corr, mean channel std, saturation fraction]
%     (Finding 5: the "valid" class is DEFINED by correlation, so if this
%      baseline matches the LSTM, the LSTM learned nothing new)
%
% Still-open caveat (Finding 3, not fixed here): the invalid class is
% dominated by railed/saturated data, so even the honest number overstates
% real pattern discrimination. A fair invalid class = clean-but-uncorrelated
% windows; that needs re-running the Python extractor.
%
% Required: Deep Learning Toolbox + Statistics and Machine Learning Toolbox.
% Run from the handoff root (folder containing classification_segments_output/).

clear; clc; close all; rng(0);

%% Configuration
labelSummaryFile = fullfile("classification_segments_output", "combined_label_summary.csv");
outputDir = "matlab_classifier_output_v2";

featureKind   = "Z";
sampleRateHz  = 200;
windowSeconds = 2.0;
strideSeconds = 0.25;
maxEpochs     = 80;
trainLSTM     = true;    % false = logistic baseline only (fast)

numFeatures = 8;
windowSteps = round(windowSeconds * sampleRateHz);
strideSteps = round(strideSeconds * sampleRateHz);
if ~isfolder(outputDir); mkdir(outputDir); end

%% Build windows, grouped by source file (the unique segment key)
summary = readtable(labelSummaryFile, TextType="string", VariableNamingRule="preserve");
featureCols = getFeatureColumns(featureKind);

allX = {};            % each cell: numFeatures x windowSteps
allY = [];            % 1 = valid_pattern, 0 = invalid_or_noise
allGroup = strings(0, 1);   % source file path (split key)
allFeat = [];         % nWindows x 3 baseline features

for i = 1:height(summary)
    segmentPath = string(summary.file(i));
    classId = summary.class_id(i);     % 1 valid, 0 invalid

    segment = readtable(segmentPath, TextType="string", VariableNamingRule="preserve");
    [~, values] = resampleSegment(segment, featureCols, sampleRateHz);
    if isempty(values) || size(values, 1) < windowSteps; continue; end

    for startIdx = 1:strideSteps:(size(values, 1) - windowSteps + 1)
        win = values(startIdx:startIdx + windowSteps - 1, :).';   % 8 x windowSteps
        allX{end+1, 1} = win;                 %#ok<AGROW>
        allY(end+1, 1) = classId;             %#ok<AGROW>
        allGroup(end+1, 1) = segmentPath;     %#ok<AGROW>
        allFeat(end+1, :) = windowFeatures(win);  %#ok<AGROW>
    end
end

groups = unique(allGroup);
fprintf("Windows: %d (valid=%d invalid=%d) across %d segments\n", ...
    numel(allY), sum(allY == 1), sum(allY == 0), numel(groups));

%% Leave-one-segment-out CV (predictions pooled across folds)
yTrueAll = []; yPredLSTM = []; yPredLogit = [];

for g = 1:numel(groups)
    testMask = allGroup == groups(g);
    trainMask = ~testMask;
    if numel(unique(allY(trainMask))) < 2; continue; end   % need both classes to train

    % balance training windows: subsample majority class to minority count
    trainIdx = balanceIdx(find(trainMask), allY, rng_local());

    % ---- logistic baseline ----
    mdl = fitclinear(allFeat(trainIdx, :), allY(trainIdx), Learner="logistic");
    yPredLogit = [yPredLogit; predict(mdl, allFeat(testMask, :))]; %#ok<AGROW>

    % ---- LSTM ----
    if trainLSTM
        XTr = allX(trainIdx);
        YTr = categorical(allY(trainIdx), [0 1], ["invalid_or_noise", "valid_pattern"]);
        layers = [
            sequenceInputLayer(numFeatures, Name="input")
            lstmLayer(96, OutputMode="last", Name="lstm")
            dropoutLayer(0.2, Name="dropout")
            fullyConnectedLayer(2, Name="scores")
            softmaxLayer(Name="softmax")
            classificationLayer(Name="classification")
        ];
        options = trainingOptions("adam", MaxEpochs=maxEpochs, MiniBatchSize=16, ...
            InitialLearnRate=1e-3, GradientThreshold=1, Shuffle="every-epoch", ...
            Verbose=false, Plots="none");
        net = trainNetwork(XTr, YTr, layers, options);
        pred = classify(net, allX(testMask), MiniBatchSize=1);
        yPredLSTM = [yPredLSTM; double(pred == "valid_pattern")]; %#ok<AGROW>
    end

    yTrueAll = [yTrueAll; allY(testMask)]; %#ok<AGROW>
    fprintf("Held out %s (%d win, class %d)\n", ...
        shortname(groups(g)), sum(testMask), unique(allY(testMask)));
end

%% Report (pooled over all held-out segments)
fprintf("\n=== Leave-one-segment-out classification (pooled) ===\n");
reportClf("Logistic baseline [corr,std,sat]", yTrueAll, yPredLogit);
if trainLSTM
    reportClf("LSTM", yTrueAll, yPredLSTM);
end
fprintf("\nCompare against the original within-recording leaky number: 97.6%%.\n");

%% Save
T = table(yTrueAll, yPredLogit, VariableNames=["true", "pred_logistic"]);
if trainLSTM && numel(yPredLSTM) == numel(yTrueAll)
    T.pred_lstm = yPredLSTM;
end
writetable(T, fullfile(outputDir, "classifier_loso_predictions.csv"));
fprintf("Done. Outputs in %s\n", outputDir);

%% Local Functions
function cols = getFeatureColumns(featureKind)
    channels = "EXG Channel " + string(0:7);
    switch featureKind
        case "Z";           suffix = " Z";
        case "Centered uV";  suffix = " Centered uV";
        case "Cleaned uV";   suffix = " Cleaned uV";
        otherwise; error("Unsupported feature kind: %s", featureKind);
    end
    cols = channels + suffix;
end

function [tUniform, valuesUniform] = resampleSegment(segment, featureCols, sampleRateHz)
    t = segment.t_rel_s;
    values = zeros(height(segment), numel(featureCols));
    for c = 1:numel(featureCols)
        values(:, c) = segment.(featureCols(c));
    end
    valid = isfinite(t) & all(isfinite(values), 2);
    t = t(valid); values = values(valid, :);
    [t, uniqueIdx] = unique(t, "stable");
    values = values(uniqueIdx, :);
    if numel(t) < 3; tUniform = []; valuesUniform = []; return; end
    dt = 1 / sampleRateHz;
    tUniform = (t(1):dt:t(end)).';
    valuesUniform = zeros(numel(tUniform), size(values, 2));
    for c = 1:size(values, 2)
        valuesUniform(:, c) = interp1(t, values(:, c), tUniform, "linear");
    end
end

function f = windowFeatures(win)
    % win: 8 x steps  ->  [mean pairwise corr, mean channel std, saturation frac]
    c = corrcoef(win.');
    iu = triu(true(8), 1);
    meanCorr = mean(c(iu), "omitnan");
    meanStd = mean(std(win, 0, 2));
    satFrac = mean(abs(win(:)) > 4);   % |z|>4 ~ saturated/railed in robust-z space
    f = [meanCorr, meanStd, satFrac];
end

function idx = balanceIdx(idxPool, allY, seed)
    rng(seed);
    yPool = allY(idxPool);
    pos = idxPool(yPool == 1); neg = idxPool(yPool == 0);
    n = min(numel(pos), numel(neg));
    pos = pos(randperm(numel(pos), n));
    neg = neg(randperm(numel(neg), n));
    idx = [pos; neg];
    idx = idx(randperm(numel(idx)));
end

function s = rng_local()
    s = 0;   % fixed seed for reproducible balancing
end

function reportClf(name, yTrue, yPred)
    if numel(yPred) ~= numel(yTrue) || isempty(yPred)
        fprintf("  %-34s (not run)\n", name); return;
    end
    acc = mean(yPred == yTrue);
    tp = sum(yPred == 1 & yTrue == 1); fp = sum(yPred == 1 & yTrue == 0);
    tn = sum(yPred == 0 & yTrue == 0); fn = sum(yPred == 0 & yTrue == 1);
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1);
    bal = (sens + spec) / 2;
    fprintf("  %-34s acc=%.1f%%  balanced=%.1f%%  (sens=%.1f%% spec=%.1f%%)\n", ...
        name, acc * 100, bal * 100, sens * 100, spec * 100);
end

function s = shortname(p)
    [~, n, e] = fileparts(p);
    s = n + e;
end
