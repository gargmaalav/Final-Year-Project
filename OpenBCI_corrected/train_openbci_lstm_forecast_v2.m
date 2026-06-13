%% OpenBCI 8-Channel LSTM Forecasting - CORRECTED (review findings 5 & 6)
% Changes vs train_openbci_lstm_forecast.m:
%   - Leave-one-segment-out CV instead of a single fixed test segment 10
%     (Finding 6: one RMSE from one segment is an anecdote, not an estimate)
%   - Persistence + linear-extrapolation baselines so the RMSE is
%     interpretable (Finding 5: 0.1945 means nothing with no reference;
%     on a smooth <1 Hz signal "repeat the last value" is a strong baseline)
%   - Per-channel RMSE and across-fold mean +/- std (Finding 6)
%
% Read the numbers like this: if the LSTM mean RMSE does NOT clearly beat
% persistence, the LSTM is not earning its keep on this data.
%
% Required: Deep Learning Toolbox. Run from the handoff root (the folder
% that contains manual_segments_output/).

clear; clc; close all; rng(0);

%% Configuration
segmentDir   = fullfile("manual_segments_output", "segments");
summaryFile  = fullfile("manual_segments_output", "manual_accepted_segments_with_times.csv");
outputDir    = "matlab_lstm_output_v2";

featureKind     = "Z";
sampleRateHz    = 200;
inputSeconds    = 2.0;
forecastSeconds = 0.5;
strideSeconds   = 0.25;
maxEpochs       = 80;     % reduced from 150: LOSO trains one net per fold
trainLSTM       = true;   % set false for baselines only (fast, no toolbox needed)

numFeatures   = 8;
inputSteps    = round(inputSeconds * sampleRateHz);
forecastSteps = round(forecastSeconds * sampleRateHz);
numResponses  = numFeatures * forecastSteps;
strideSteps   = round(strideSeconds * sampleRateHz);

if ~isfolder(outputDir); mkdir(outputDir); end

%% Build per-segment windows (kept separate so CV never mixes a segment)
summary = readtable(summaryFile, TextType="string", VariableNamingRule="preserve");
featureCols = getFeatureColumns(featureKind);

segIds = [];
segX = {};   % segX{s} = cell array of input windows for segment s
segY = {};   % segY{s} = (nWindows x numResponses) targets for segment s

for i = 1:height(summary)
    segmentId = summary.segment_id(i);
    label = summary.label(i);
    fileName = sprintf("manual_segment_%03d_%s.csv", segmentId, label);
    segmentPath = fullfile(segmentDir, fileName);

    segment = readtable(segmentPath, TextType="string", VariableNamingRule="preserve");
    [tU, values] = resampleSegment(segment, featureCols, sampleRateHz);

    [Xs, Ys] = makeForecastWindows(tU, values, inputSteps, forecastSteps, strideSteps);
    if isempty(Xs); continue; end

    segIds(end+1) = segmentId; %#ok<AGROW>
    segX{end+1} = Xs;          %#ok<AGROW>
    segY{end+1} = Ys;          %#ok<AGROW>
end

nSeg = numel(segX);
fprintf("Segments with windows: %d\n", nSeg);

%% Leave-one-segment-out cross-validation
foldRmseLSTM = nan(nSeg, 1); chRmseLSTM = nan(nSeg, numFeatures);
foldRmsePers = nan(nSeg, 1); chRmsePers = nan(nSeg, numFeatures);
foldRmseLin  = nan(nSeg, 1); chRmseLin  = nan(nSeg, numFeatures);

for k = 1:nSeg
    testX = segX{k};
    testY = segY{k};

    % ---- baselines (no training) ----
    [pPred, lPred] = forecastBaselines(testX, inputSteps, forecastSteps, numFeatures);
    [foldRmsePers(k), chRmsePers(k, :)] = rmseByChannel(pPred, testY, numFeatures);
    [foldRmseLin(k),  chRmseLin(k, :)]  = rmseByChannel(lPred, testY, numFeatures);

    % ---- LSTM trained on the other segments ----
    if trainLSTM
        trainIdx = setdiff(1:nSeg, k);
        XTrain = vertcat(segX{trainIdx});
        YTrain = vertcat(segY{trainIdx});

        layers = [
            sequenceInputLayer(numFeatures, Name="input")
            lstmLayer(128, OutputMode="last", Name="lstm")
            dropoutLayer(0.2, Name="dropout")
            fullyConnectedLayer(numResponses, Name="forecast")
            regressionLayer(Name="regression")
        ];
        options = trainingOptions("adam", ...
            MaxEpochs=maxEpochs, MiniBatchSize=16, InitialLearnRate=1e-3, ...
            GradientThreshold=1, Shuffle="every-epoch", Verbose=false, Plots="none");

        net = trainNetwork(XTrain, YTrain, layers, options);
        YPred = predict(net, testX, MiniBatchSize=1);
        [foldRmseLSTM(k), chRmseLSTM(k, :)] = rmseByChannel(YPred, testY, numFeatures);
    end

    fprintf("Fold %2d (seg %2d, %4d win): LSTM=%.4f  persistence=%.4f  linear=%.4f\n", ...
        k, segIds(k), numel(testX), foldRmseLSTM(k), foldRmsePers(k), foldRmseLin(k));
end

%% Report
fprintf("\n=== Leave-one-segment-out forecast RMSE (%s space) ===\n", featureKind);
reportFold("LSTM", foldRmseLSTM);
reportFold("Persistence (repeat last frame)", foldRmsePers);
reportFold("Linear extrapolation", foldRmseLin);

fprintf("\nPer-channel RMSE (mean over folds):\n");
fprintf("  ch :   LSTM   persist  linear\n");
for c = 1:numFeatures
    fprintf("  %2d : %7.4f %7.4f %7.4f\n", c - 1, ...
        mean(chRmseLSTM(:, c), "omitnan"), mean(chRmsePers(:, c), "omitnan"), ...
        mean(chRmseLin(:, c), "omitnan"));
end

%% Save
results = table(segIds(:), foldRmseLSTM, foldRmsePers, foldRmseLin, ...
    VariableNames=["segment_id", "rmse_lstm", "rmse_persistence", "rmse_linear"]);
writetable(results, fullfile(outputDir, "forecast_loso_results.csv"));
save(fullfile(outputDir, "forecast_loso.mat"), ...
    "foldRmseLSTM", "foldRmsePers", "foldRmseLin", ...
    "chRmseLSTM", "chRmsePers", "chRmseLin", "segIds");
fprintf("\nDone. Outputs in %s\n", outputDir);

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

function [X, Y] = makeForecastWindows(tUniform, values, inputSteps, forecastSteps, strideSteps)
    totalSteps = inputSteps + forecastSteps;
    X = {}; Y = [];
    if isempty(values) || size(values, 1) < totalSteps; return; end
    row = 0;
    for startIdx = 1:strideSteps:(size(values, 1) - totalSteps + 1)
        inputEndIdx = startIdx + inputSteps - 1;
        fStart = inputEndIdx + 1; fEnd = inputEndIdx + forecastSteps;
        row = row + 1;
        X{row, 1} = values(startIdx:inputEndIdx, :).'; %#ok<AGROW>
        yF = values(fStart:fEnd, :).';
        Y(row, :) = yF(:).'; %#ok<AGROW>
    end
end

function [pPred, lPred] = forecastBaselines(testX, inputSteps, forecastSteps, numFeatures)
    % persistence  = repeat the last observed frame across the horizon
    % linear       = per-channel least-squares line fit over the input,
    %                extrapolated into the horizon
    n = numel(testX);
    R = numFeatures * forecastSteps;
    pPred = zeros(n, R); lPred = zeros(n, R);
    tIn = 1:inputSteps;
    tFut = inputSteps + (1:forecastSteps);
    for w = 1:n
        xw = testX{w};               % numFeatures x inputSteps
        pMat = repmat(xw(:, end), 1, forecastSteps);
        pPred(w, :) = pMat(:).';
        lMat = zeros(numFeatures, forecastSteps);
        for c = 1:numFeatures
            pc = polyfit(tIn, xw(c, :), 1);
            lMat(c, :) = polyval(pc, tFut);
        end
        lPred(w, :) = lMat(:).';
    end
end

function [rmseAll, rmseCh] = rmseByChannel(pred, trueY, numFeatures)
    % responses are flattened column-major as [ch x step], so channel c
    % occupies columns c, c+8, c+16, ...
    E = pred - trueY;
    rmseAll = sqrt(mean(E(:).^2));
    rmseCh = zeros(1, numFeatures);
    for c = 1:numFeatures
        Ec = E(:, c:numFeatures:end);
        rmseCh(c) = sqrt(mean(Ec(:).^2));
    end
end

function reportFold(name, v)
    v = v(~isnan(v));
    if isempty(v)
        fprintf("  %-32s (not run)\n", name); return;
    end
    fprintf("  %-32s mean=%.4f  std=%.4f  min=%.4f  max=%.4f  (%d folds)\n", ...
        name, mean(v), std(v), min(v), max(v), numel(v));
end
