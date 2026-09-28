% Folder structure:
%   +PredictiveModels/LLM_EKF_Based_Predictor1.m

function LLM_EKF_Based_Predictor1(symbol)
    %% ---------------------- Reproducibility Seed -----------------------
    baseSeed = 42;
    rng(baseSeed, 'twister');
    
    %% ===================================================================
    %  ADVANCED LLM + EKF SMOOTHER + MULTI-STEP LSTM FORECASTING SYSTEM
    %  Namespace: +PredictiveModels
    % ===================================================================

    %% ===================================================================
    %     PART 1 — Tiingo OHLCV Ingestion (Bulletproof I/O Gateway)
    %% ===================================================================
    fprintf("📡 Downloading Market Telemetry from Tiingo for ticker: %s…\n", upper(symbol));
    apiKey = "ec32ec3a512d8306f6ff89346618ff705fc52b08";
    localFile = sprintf('%s_Tiingo_cache.csv', upper(symbol));

    % FIX: the cache was trusted forever once it existed -- rerunning
    % this script for the same ticker on a LATER day silently kept
    % feeding the EKF/BiLSTM pipeline yesterday's (or last week's) prices
    % with no warning at all, since nothing ever re-validated freshness.
    % refreshCacheIfStale (defined below) subsumes the old corrupt/
    % too-short check AND deletes the cache if its newest row is more
    % than a few calendar days old, so the download block right below
    % naturally re-fetches current data instead of reusing stale rows.
    refreshCacheIfStale(localFile, 'date', 4);

    if ~exist(localFile, 'file')
        % FIX: this sprintf format string had no %s placeholders at all, so
        % it always evaluated to the literal text 'https://tiingo.com' --
        % Tiingo's marketing homepage, not an API call -- regardless of
        % ticker or apiKey. That's why this failed for every ticker, not
        % just low-history/recent listings: websave() either errors on
        % that response or writes something readtable can't parse as
        % OHLCV data, so localFile never ends up populated with real data
        % and the "recent listing" framing in the fallback message is
        % misleading -- the request itself was never valid.
        %
        % Primary request: last 5 years of daily OHLCV as CSV.
        startDatePrimary = datestr(datetime('today') - calyears(5), 'yyyy-mm-dd');
        url = sprintf(['https://api.tiingo.com/tiingo/daily/%s/prices?' ...
            'startDate=%s&format=csv&columns=open,high,low,close,volume&token=%s'], ...
            lower(symbol), startDatePrimary, apiKey);
        try
            websave(localFile, url);
            downloadedTable = readtable(localFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
            if isempty(downloadedTable) || height(downloadedTable) < 5
                error('Empty or too-short response for %s', symbol);
            end
        catch ME
            if exist(localFile, 'file'), delete(localFile); end
            fprintf("ℹ️ Asset broad query returned network shift action (%s).\n", ME.message);
            fprintf("🔄 Activating low-history optimization gateway for recent listings...\n");
            % Fallback: drop startDate entirely so a recently-listed ticker
            % (limited history) still returns whatever Tiingo has, instead
            % of repeating the exact same (broken) request as the primary.
            fallbackUrl = sprintf(['https://api.tiingo.com/tiingo/daily/%s/prices?' ...
                'format=csv&columns=open,high,low,close,volume&token=%s'], ...
                lower(symbol), apiKey);
            try
                websave(localFile, fallbackUrl);
                downloadedTable = readtable(localFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
            catch fallbackME
                if exist(localFile, 'file'), delete(localFile); end
                error("❌ Tiingo API Network Handshake rejected. Reason: %s", fallbackME.message);
            end
        end
    else
        downloadedTable = readtable(localFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
    end

    %% ---------------------- Validate Columns --------------------------
    requiredCols = {'date','open','high','low','close','volume'};
    rawCols = lower(downloadedTable.Properties.VariableNames);
    missing = setdiff(requiredCols, rawCols);
    if ~isempty(missing)
        error("❌ Tiingo CSV missing required columns: %s", strjoin(missing, ', '));
    end

    %% ---------------------- Parse & Clean OHLCV Arrays ----------------
    dates       = datetime(downloadedTable.date, 'InputFormat','yyyy-MM-dd');
    openPrices  = double(downloadedTable.open);
    highPrices  = double(downloadedTable.high);
    lowPrices   = double(downloadedTable.low);
    closePrices = double(downloadedTable.close);
    volumes     = double(downloadedTable.volume);

    validMask = ~(isnan(openPrices) | isnan(highPrices) | isnan(lowPrices) | isnan(closePrices) | isnan(volumes) | isnat(dates));
    dates = dates(validMask); openPrices = openPrices(validMask); highPrices = highPrices(validMask);
    lowPrices = lowPrices(validMask); closePrices = closePrices(validMask); volumes = volumes(validMask);

    if numel(closePrices) < 60
        error("❌ Not enough data rows (%d found, <60 required) to sync framework for %s.", numel(closePrices), symbol);
    end

    %% ---------------------- Build Master Timetable ---------------------
    marketTT = timetable(openPrices, closePrices, highPrices, lowPrices, volumes, ...
        'RowTimes', dates, 'VariableNames', {'Open','Close','High','Low','Volume'});
    marketTT.Properties.DimensionNames{1} = 'Time';
    marketTT = sortrows(marketTT, 'Time', 'ascend');
    numDays = height(marketTT);

    fprintf("📁 Ingestion pipeline synchronized. Usable rows compiled: %d\n", numDays);

    %% ===================================================================
    %  LLM TEXT SENTIMENT FLAVOR ENGINE (NEW INTEGRATION GATEWAY)
    %% ===================================================================
    fprintf("🤖 Checking for LLM Semantic Text Insights Cache...\n");
    llmFile = sprintf('%s_LLM_Sentiment_cache.csv', upper(symbol));

    % -------------------------------------------------------------------
    % ENHANCEMENT: real LLM connection. Previously "LLM_Sentiment" was
    % ALWAYS synthetic noise (0.1*sin(...) + 0.05*randn(...)) whenever
    % <TICKER>_LLM_Sentiment_cache.csv didn't already exist on disk --
    % there was no code path that ever actually called an LLM or fetched
    % real news. This block adds one: pull recent headlines from Tiingo's
    % News API (same apiKey you already have), have Claude score each
    % headline's sentiment, and write the result to the SAME cache file
    % schema (date, llm_sentiment_score) the existing read-path below
    % already expects -- so nothing downstream needs to change.
    %
    % SECURITY: the key is read from the ANTHROPIC_API_KEY environment
    % variable rather than hardcoded here, so it never lives in this
    % file (source control, shared folders, etc). Set it once, however
    % suits your workflow:
    %   - Inside MATLAB, for just this session:
    %       setenv('ANTHROPIC_API_KEY', 'sk-ant-...')
    %   - macOS/Linux shell, before launching MATLAB:
    %       export ANTHROPIC_API_KEY="sk-ant-..."
    %   - Windows PowerShell (persists for future sessions):
    %       setx ANTHROPIC_API_KEY "sk-ant-..."
    % Get a key at https://console.anthropic.com/ . With nothing set,
    % this falls straight through to cache/synthetic sentiment as before.
    % -------------------------------------------------------------------
    claudeApiKey = string(getenv('ANTHROPIC_API_KEY'));
    haveLiveLLM = strlength(strtrim(claudeApiKey)) > 0;

    % -------------------------------------------------------------------
    % ENHANCEMENT: track WHICH sentiment path actually ran this session,
    % so the dashboard can tell the user plainly whether the LLM Sentiment
    % feature feeding the BiLSTM is real news-derived scoring, a cache
    % from an earlier live run, or the synthetic sine+noise placeholder.
    % Previously this was invisible -- the table showed a sentiment
    % NUMBER regardless of source, with no way to tell which case produced
    % it short of reading console output line-by-line.
    % -------------------------------------------------------------------
    sentimentSourceLabel = "";  % filled in below, used by the Part 7 table

    if haveLiveLLM
        try
            fprintf("🧠 Fetching recent news for %s and scoring sentiment via Claude...\n", upper(symbol));
            newsItems = fetchTiingoNews(symbol, apiKey, 50);
            llmScoresTable = scoreNewsSentimentWithClaude(newsItems, claudeApiKey);
            writetable(llmScoresTable, llmFile);
            fprintf("✅ Live LLM sentiment refreshed and cached to %s (%d dated entries).\n", ...
                llmFile, height(llmScoresTable));
            sentimentSourceLabel = "🟢 LIVE (Claude-scored news, refreshed this run)";
        catch llmFetchErr
            fprintf("⚠️ Live LLM sentiment fetch failed (%s).\n", llmFetchErr.message);
            fprintf("   Falling back to existing cache file / synthetic context below.\n");
        end
    else
        fprintf("ℹ️ ANTHROPIC_API_KEY not set in the environment -- skipping live LLM sentiment scoring.\n");
    end

    if exist(llmFile, 'file')
        try
            llmTable = readtable(llmFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
            llmDates = datetime(llmTable.date, 'InputFormat', 'yyyy-MM-dd');
            llmScores = double(llmTable.llm_sentiment_score);
            llmTT = timetable(llmScores, 'RowTimes', llmDates, 'VariableNames', {'LLMSentiment'});
            llmTT.Properties.DimensionNames{1} = 'Time';
            alignedLLM = retime(llmTT, marketTT.Time, 'previous');
            LLM_Sentiment = fillmissing(alignedLLM.LLMSentiment, 'constant', 0);
            fprintf("✅ LLM Qualitative Sentiment text context synchronized successfully.\n");
            if sentimentSourceLabel == ""
                % File existed and parsed fine, but this run did NOT just
                % write it live above -- it's a cache from a prior run.
                cacheAgeDays = days(datetime('today') - max(llmDates));
                sentimentSourceLabel = sprintf("🟡 CACHED (from prior live run, %.0fd old)", cacheAgeDays);
            end
        catch llmErr
            fprintf("⚠️ Found LLM cache but parsing failed (%s). Defaulting to Neutral (0).\n", llmErr.message);
            LLM_Sentiment = zeros(numDays, 1);
            sentimentSourceLabel = "⚪ NEUTRAL FALLBACK (cache found but failed to parse)";
        end
    else
        fprintf("ℹ️ %s not found. Synthetic context fallback generated for model processing.\n", llmFile);
        fprintf("   (Set the ANTHROPIC_API_KEY environment variable to replace this with real news-derived sentiment.)\n");
        LLM_Sentiment = 0.1 * sin((1:numDays)' / 20) + 0.05 * randn(numDays, 1);
        sentimentSourceLabel = "🔴 SYNTHETIC (sine+noise placeholder -- not real sentiment)";
    end
    LLM_Sentiment_Smoothed = movmean(LLM_Sentiment, [2 0]);
    LLM_Sentiment_Velocity = [0; diff(LLM_Sentiment_Smoothed)];
    marketTT.LLM_Sentiment_Vel = LLM_Sentiment_Velocity;

    %% ===================================================================
    %  MARKET CONTEXT (SPY BENCHMARK)
    %% ===================================================================
    try
        spyLocalFile = 'SPY_Tiingo_cache.csv';
        % Same staleness bug as the main ticker cache above -- once this
        % file existed it was reused forever with no freshness check.
        refreshCacheIfStale(spyLocalFile, 'date', 4);
        if ~exist(spyLocalFile, 'file')
            % FIX: same missing-%s bug as the main ticker fetch -- this
            % always resolved to the bare 'https://tiingo.com' homepage,
            % so SPY's local cache was never actually populated by a real
            % API response either.
            startDateSpy = datestr(datetime('today') - calyears(5), 'yyyy-mm-dd');
            spyUrl = sprintf(['https://api.tiingo.com/tiingo/daily/spy/prices?' ...
                'startDate=%s&format=csv&columns=open,high,low,close,volume&token=%s'], ...
                startDateSpy, apiKey);
            websave(spyLocalFile, spyUrl);
        end
        spyTableRaw = readtable(spyLocalFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
        spyDatesRaw = datetime(spyTableRaw.date, 'InputFormat', 'yyyy-MM-dd');
        spyCloseRaw = double(spyTableRaw.close);
        spyValidMask = ~isnat(spyDatesRaw) & ~isnan(spyCloseRaw);

        spyTT = timetable(spyCloseRaw(spyValidMask), 'RowTimes', spyDatesRaw(spyValidMask), 'VariableNames', {'SPYClose'});
        spyTT.Properties.DimensionNames{1} = 'Time';
        spyTT = sortrows(spyTT, 'Time', 'ascend');

        alignedSPY = retime(spyTT, marketTT.Time, 'previous');
        spyCloseAligned = fillmissing(alignedSPY.SPYClose, 'nearest');
        MarketReturn = [0; diff(spyCloseAligned) ./ max(spyCloseAligned(1:end-1), 1e-6)];
    catch
        MarketReturn = zeros(numDays, 1);
    end

    %% ===================================================================
    %     CORE TECHNICAL EXTRACTORS & STRUCTURAL FEATURES
    %% ===================================================================
    h_Open = marketTT.Open; h_Close = marketTT.Close; h_High = marketTT.High; h_Low = marketTT.Low; h_Vol = marketTT.Volume;

    localRetCalculated = [0; diff(h_Close) ./ max(h_Close(1:end-1), 1e-6)];

    h_High_Clean = max(h_High, 1e-6); h_Low_Clean = max(h_Low, 1e-6);
    parkinsonRaw = (log(h_High_Clean ./ h_Low_Clean)).^2 / (4 * log(2));
    ParkinsonVol = sqrt(movmean(parkinsonRaw, [19 0], 'omitnan')); 

    try
        xlvLocalFile = 'XLV_Tiingo_cache.csv';
        % Same staleness bug as the main ticker cache above.
        refreshCacheIfStale(xlvLocalFile, 'date', 4);
        % FIX: unlike the SPY block above, this never attempted a download
        % at all -- it only ever tried to read a local file that (on a
        % fresh machine, or any ticker other than one already cached)
        % simply doesn't exist, silently falling into the catch below and
        % substituting MarketReturn for SectorRelativeStrength every time.
        % Added a real fetch so it behaves the same way SPY does.
        if ~exist(xlvLocalFile, 'file')
            startDateXlv = datestr(datetime('today') - calyears(5), 'yyyy-mm-dd');
            xlvUrl = sprintf(['https://api.tiingo.com/tiingo/daily/xlv/prices?' ...
                'startDate=%s&format=csv&columns=open,high,low,close,volume&token=%s'], ...
                startDateXlv, apiKey);
            websave(xlvLocalFile, xlvUrl);
        end
        xlvTableRaw = readtable(xlvLocalFile, 'FileType', 'text', 'VariableNamingRule', 'preserve');
        xlvCloseRaw = double(xlvTableRaw.close);
        xlvCloseAligned = fillmissing(retime(timetable(datetime(xlvTableRaw.date), xlvCloseRaw), marketTT.Time, 'previous').Var1, 'nearest');
        SectorReturn = [0; diff(xlvCloseAligned) ./ max(xlvCloseAligned(1:end-1), 1e-6)];
        SectorRelativeStrength = localRetCalculated - SectorReturn;
    catch
        SectorRelativeStrength = localRetCalculated - MarketReturn; 
    end

    downDayVolume = zeros(size(h_Vol));
    downDayVolume(localRetCalculated < 0) = h_Vol(localRetCalculated < 0);
    VPM_Ratio = movmean(downDayVolume, [4 0], 'omitnan') ./ max(movmean(h_Vol, [19 0], 'omitnan'), 1e-6);

    dailyTrueRange = max([h_High - h_Low, abs(h_High - [h_Close(1); h_Close(1:end-1)]), abs(h_Low - [h_Close(1); h_Close(1:end-1)])], [], 2);
    RangeExpansionRatio = dailyTrueRange ./ max(movmean(dailyTrueRange, [19 0], 'omitnan'), 1e-6);

    dPrice = [0; diff(h_Close)]; gains = max(dPrice, 0); losses = max(-dPrice, 0);
    rsiCalculated = 100 - (100 ./ (1 + movmean(gains, [13 0]) ./ max(movmean(losses, [13 0]), 1e-6)));

    %% ===================================================================
    %  ENHANCEMENT: ADAPTIVE ASYMMETRIC EKF WITH MOMENTUM EXHAUSTION
    %% ===================================================================
    fprintf("⚡ Injecting Advanced Extended Kalman Filter Smoother Tracking Matrices...\n");
    
    N = numDays;
    X_est = zeros(2, N); 
    P_est = cell(N, 1);
    
    X_est(:, 1) = [h_Close(1); 0];

    % -------------------------------------------------------------------
    % GENERALIZATION: everything in this block used to be a fixed
    % absolute-dollar constant (P_curr, Q, R_meas, the $25.00 momentum
    % trigger, the $10.00 damping divisor) -- values that only make sense
    % for a stock trading in roughly the same $20-30 band it was tuned
    % on. A $2 microcap or a $500 large-cap would get wildly wrong
    % noise/damping behavior from the same numbers. Everything below is
    % now derived from the ticker's OWN recent ATR (dollar volatility)
    % and trailing 1-year price range, so the filter's assumptions scale
    % with whatever instrument you actually pass in.
    % -------------------------------------------------------------------
    recentATR = mean(dailyTrueRange(max(1, N-13):N), 'omitnan');   % ~14-day ATR, in $
    recentATR = max(recentATR, 1e-3 * max(h_Close(N), 1));         % floor for very quiet names

    lookbackRange = min(252, N);                                   % ~1 trading year, or all available history
    rollingLow  = movmin(h_Close, [lookbackRange-1 0]);
    rollingHigh = movmax(h_Close, [lookbackRange-1 0]);
    rollingSpan = max(rollingHigh - rollingLow, 1e-6);

    P_curr = [recentATR^2, 0.0; 0.0, 0.1 * recentATR^2];
    P_est{1} = P_curr;                     
    
    dt = 1.0;
    Q = [0.02 * recentATR^2, 0.01 * recentATR^2; 0.01 * recentATR^2, 0.05 * recentATR^2];
    R_meas = 0.25 * recentATR^2;

    for t = 2:N
        % "Value zone" ceiling for momentum-exhaustion damping: bottom 25%
        % of the TRAILING (point-in-time, no look-ahead) 1-year range as
        % of t-1, replacing the fixed $25.00 cutoff. The old $10.00
        % divisor (how fast damping ramps in below the trigger) is
        % likewise replaced with 10% of the trailing range span.
        valueZoneCeiling = rollingLow(t-1) + 0.25 * rollingSpan(t-1);
        dampingSpan = max(0.10 * rollingSpan(t-1), 1e-6);

        velDamping = 1.0;
        if X_est(2, t-1) < 0 && h_Close(t-1) < valueZoneCeiling
            velDamping = max(0.40, 1.0 - (valueZoneCeiling - h_Close(t-1)) / dampingSpan);
        end
        F = [1, dt; 0, velDamping]; 
        
        X_pred = F * X_est(:, t-1);
        P_pred = F * P_curr * F' + Q;
        
        % Dynamic transaction impedance weighting
        % Dynamic transaction impedance weighting
        volFactor = max(h_Vol(t) / max(mean(h_Vol(max(1, t-20):t)), 1e-6), 0.1);
        R_adaptive = R_meas * (1.0 / volFactor); 
        
        H = [1 0]; 
        y_innov = h_Close(t) - X_pred(1); 
        
        S = H * P_pred * H' + R_adaptive;
        K_gain = (P_pred * H') / S;
        
        X_est(:, t) = X_pred + K_gain * y_innov;
        P_curr = (eye(2) - K_gain * H) * P_pred;
        P_est{t} = P_curr;
    end
    
    % Backward RTS (Rauch-Tung-Striebel) Smoothing Sweep to remove lag
    X_smooth = zeros(2, N);
    X_smooth(:, end) = X_est(:, end);
    P_smooth = P_est;
    
    for t = (N-1):-1:1
        % Same generalization as the forward pass above: value-zone
        % ceiling from the trailing (point-in-time) 1-year range at t,
        % instead of the fixed $25.00/$10.00 constants.
        valueZoneCeiling_b = rollingLow(t) + 0.25 * rollingSpan(t);
        dampingSpan_b = max(0.10 * rollingSpan(t), 1e-6);

        velDamping_b = 1.0;
        if X_est(2, t) < 0 && h_Close(t) < valueZoneCeiling_b
            velDamping_b = max(0.40, 1.0 - (valueZoneCeiling_b - h_Close(t)) / dampingSpan_b);
        end
        F_b = [1, dt; 0, velDamping_b];
        
        X_pred_next = F_b * X_est(:, t);
        P_pred_next = F_b * P_est{t} * F_b' + Q; 
        
        C_gain = (P_est{t} * F_b') / P_pred_next;
        X_smooth(:, t) = X_est(:, t) + C_gain * (X_smooth(:, t+1) - X_pred_next);
        P_smooth{t}  = P_est{t} + C_gain * (P_smooth{t+1} - P_pred_next) * C_gain';
    end
    
    marketTT.EKF_TruePrice = X_smooth(1, :).';
    marketTT.EKF_Velocity  = X_smooth(2, :).';
    marketTT.EKF_Residual  = h_Close - marketTT.EKF_TruePrice;

    %% ---------------------- Append To Core Matrix ----------------------
    marketTT.ParkinsonVol = ParkinsonVol;
    marketTT.SectorRS     = SectorRelativeStrength;
    marketTT.VPMRatio     = VPM_Ratio;
    marketTT.RangeExp     = RangeExpansionRatio;
    marketTT.LLM_Sentiment = LLM_Sentiment_Smoothed;
    marketTT.Returns       = localRetCalculated;
    marketTT.RSI           = rsiCalculated;

    %% ===================================================================
    %    PART 4 — Unified Feature Matrix & Horizon Expansion (45 DAYS)
    %% ===================================================================
    horizonAhead = 45; 
    targetMatrix = zeros(numDays, horizonAhead);
    
    for t = 1:(numDays - horizonAhead)
        for h = 1:horizonAhead
            targetMatrix(t, h) = (h_Close(t + h) - h_Close(t)) / max(h_Close(t), 1e-6);
        end
    end
    % Handle tail padding safely
    for t = (numDays - horizonAhead + 1):numDays
        targetMatrix(t, :) = repmat(targetMatrix(numDays - horizonAhead, end), 1, horizonAhead);
    end

    varNames = marketTT.Properties.VariableNames;
    for k = 1:numel(varNames), marketTT.(varNames{k}) = fillmissing(marketTT.(varNames{k}), 'nearest'); end

    featureList = {'Returns', 'RSI', 'LLM_Sentiment', 'ParkinsonVol', 'SectorRS', ...
                   'VPMRatio', 'RangeExp', 'EKF_TruePrice', 'EKF_Velocity', 'EKF_Residual'};
    numFeatures = length(featureList);
    
    %% ---------------------- Non-Leaking Window Normalization -----------
    featureMatrix = marketTT{:, featureList};
    featureNorm = zeros(size(featureMatrix));
    [numRows, numFeat] = size(featureMatrix);
    baseLookbackWindow = 60; 

    for r = 1:numRows
        if r <= baseLookbackWindow
            subMat = featureMatrix(1:baseLookbackWindow, :);
        else
            subMat = featureMatrix(r - baseLookbackWindow : r - 1, :);
        end
        mu_roll = mean(subMat, 1);
        sigma_roll = std(subMat, 0, 1);
        
        for f = 1:numFeat
            featureNorm(r, f) = (featureMatrix(r, f) - mu_roll(f)) / max(sigma_roll(f), 1e-6);
        end
    end

    lookback = 10; 
    totalSamples = height(marketTT) - lookback - horizonAhead + 1;
    X_Cell = cell(totalSamples, 1); 
    Y_MultiStep = zeros(totalSamples, horizonAhead);

    for i = 1:totalSamples
        X_Cell{i}        = featureNorm(i : i + lookback - 1, :).';
        Y_MultiStep(i,:) = targetMatrix(i + lookback - 1, :);
    end

    fprintf('🎯 EKF-Feature Matrix expanded to true %d-day runway targets.\n', horizonAhead);

    %% ===================================================================
    %         PART 5 — Vector Multi-Step BiLSTM Model Architecture
    %% ===================================================================
    netArch = [
        sequenceInputLayer(numFeatures, 'Name', 'input')
        
        bilstmLayer(128, 'OutputMode', 'sequence', 'Name', 'bilstm_1')
        dropoutLayer(0.30, 'Name', 'drop_1')
        
        bilstmLayer(64, 'OutputMode', 'last', 'Name', 'bilstm_2')
        dropoutLayer(0.20, 'Name', 'drop_2')
        
        fullyConnectedLayer(horizonAhead, 'Name', 'fc_multistep')
        regressionLayer('Name', 'output')
    ];

    opts = trainingOptions('adam', ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 5, ...
        'LearnRateDropFactor', 0.7, ...
        'ExecutionEnvironment', 'auto', ... 
        'Verbose', false, ...
        'Plots', 'none');    

    %% ---------------------- Ensemble Run Matrix ----------------------
    latestBlock = featureNorm(end - lookback + 1 : end, :).';
    numTrainRuns = 3; 
    ensemblePredictions = zeros(numTrainRuns, horizonAhead);
    
    fprintf('🏋️ Training Multi-Step Predictor Core Arrays Across Ensemble Tracks...\n');
    for runIdx = 1:numTrainRuns
        rng(baseSeed + runIdx, 'twister');
        modelRun = trainNetwork(X_Cell, Y_MultiStep, netArch, opts);
        ensemblePredictions(runIdx, :) = predict(modelRun, {latestBlock});
    end

    % Process Precision Variance Weights Across the 45-Frame Forecast Vector
    meanPred = mean(ensemblePredictions, 1);

    % FIX: the ensemble weighting below used to read:
    %   varRuns = sum((ensemblePredictions - meanPred).^2, 1) + 1e-6;   % already a 1x45 ROW vector
    %   weights = (1 ./ varRuns) ./ sum(1 ./ varRuns, 1);               % sum(X,1) on a 1-row matrix is a no-op
    %   final45FrameReturns = sum(ensemblePredictions .* weights, 1);
    % sum(X, 1) on a matrix with only ONE row returns that row unchanged
    % (there's nothing else in dimension 1 to add), so `weights` collapsed
    % to exactly 1 for every frame regardless of how much the numTrainRuns
    % runs actually agreed or disagreed. That made final45FrameReturns the
    % raw SUM of the numTrainRuns runs' predictions (~numTrainRuns-x too
    % large), not a variance-weighted average. Replaced with a
    % straightforward mean across runs.
    final45FrameReturns = meanPred;

       %% ===================================================================
    %           PART 6 — Consensus Profile Path & Sizing Overrides
    %% ===================================================================
    basePrice = h_Close(end);
    predictedPricePath = zeros(horizonAhead, 1);
    for h = 1:horizonAhead
        predictedPricePath(h) = basePrice * (1 + final45FrameReturns(h));
    end

    % AUTOMATION FIX: Dynamically anchor the start date to the calendar runtime execution date
    startDate = datetime('today');   
    futureDates = datetime.empty(0, 1); 
    curDate = startDate;
    
    % Generate the next 45 forward-looking business days (skipping weekends)
    while numel(futureDates) < horizonAhead
        curDate = curDate + caldays(1);
        if weekday(curDate) >= 2 && weekday(curDate) <= 6
            futureDates = [futureDates; curDate];
        end
    end

    currentLLMSentiment = LLM_Sentiment_Smoothed(end);

    % -------------------------------------------------------------------
    %  ENHANCEMENT: COMPUTE RECENT MACD & RSI PIVOT SIGNALS FOR ENTRY
    % -------------------------------------------------------------------
    % Dynamically derive exponential moving averages for standard 12/26/9 MACD framework
    ema12 = movmean(h_Close, [11 0], 'Endpoints', 'shrink');
    ema26 = movmean(h_Close, [25 0], 'Endpoints', 'shrink');
    macdLine = ema12 - ema26;
    signalLine = movmean(macdLine, [8 0], 'Endpoints', 'shrink');

    % Extract terminal (today's) indicator states
    currentRSI = rsiCalculated(end);
    currentMacd = macdLine(end);
    currentSignal = signalLine(end);
    
    % Track previous session status to register a valid structural crossover line
    prevMacd = macdLine(end-1);
    prevSignal = signalLine(end-1);

    % Technical Signals Evaluation Flags
    isOversold = (currentRSI <= 30.0);
    isMacdBullishCross = (prevMacd <= prevSignal) && (currentMacd > currentSignal);

    % -------------------------------------------------------------------
    %       AUTOMATED BOTTOM FISHING BUY-GRID OVERRIDES ENGINE
    % -------------------------------------------------------------------
    allocatedTargetBankroll = 25000; 
    currentMarketPrice = basePrice;

    % -------------------------------------------------------------------
    % GENERALIZATION: the grid gate/tiers used to be fixed dollar cutoffs
    % ($23.50 gate; $21.50/$22.00/$22.80 tiers) that only made sense for
    % one specific ticker's price level at the time this was written.
    % Replaced with the current price's position within its own trailing
    % 1-year range, expressed as a 0..1 percentile (0 = at the trailing
    % low, 1 = at the trailing high) -- this is scale-invariant, so a
    % $3 stock and a $300 stock get the same "how close to its own
    % recent floor is this" treatment.
    % -------------------------------------------------------------------
    gridRangeWindow = h_Close(max(1, numDays - lookbackRange + 1):numDays);
    gridRangeHigh = max(gridRangeWindow);
    gridRangeLow  = min(gridRangeWindow);
    gridRangeSpan = max(gridRangeHigh - gridRangeLow, 1e-6);
    pctOfTrailingRange = (currentMarketPrice - gridRangeLow) / gridRangeSpan;

    % -------------------------------------------------------------------
    % ENHANCEMENT: COMPUTED RESISTANCE (AWARENESS) & SUPPORT (STOP-LOSS
    % GATING) LEVELS
    % -------------------------------------------------------------------
    % Fractal swing-pivot detector over the same trailing 1yr window used
    % for the grid logic above: a bar is a swing high/low if it is the
    % strict extreme of its own +/-3-bar neighborhood. Nearby pivots are
    % then clustered using an ATR-scaled tolerance (so this generalizes
    % across price levels/volatility regimes the same way recentATR
    % already does elsewhere), and the "strength" of a level is how many
    % pivots fell inside its cluster (touch count / confluence).
    [resistanceLevel, supportLevel, resistanceStrength, supportStrength] = ...
        computeSupportResistance(h_High, h_Low, currentMarketPrice, recentATR, lookbackRange);

    fprintf('📐 Structural Levels — Resistance: $%.2f (%d touches, trailing %dd) | Support: $%.2f (%d touches)\n', ...
        resistanceLevel, resistanceStrength, lookbackRange, supportLevel, supportStrength);

    % Optional stop-loss trigger buffer: place the actual stop slightly
    % below the computed support (half an ATR) rather than exactly at it,
    % to avoid getting stopped out by routine noise wicking the support
    % level itself.
    suggestedStopLoss = supportLevel - 0.5 * recentATR;

    % Core Value Guardrails: Active grid zone target filter
    % (was: currentMarketPrice <= 23.50 -- now: bottom 35% of trailing range)
    if pctOfTrailingRange <= 0.35
        fprintf('\n🎣 Deep Value Discount Alert: Market price ($%.2f, %.0f%% of trailing 1yr range) inside active grid zones.\n', ...
            currentMarketPrice, pctOfTrailingRange * 100);
        fprintf('📊 Technical Screening: Current RSI = %.1f (Oversold: %s) | MACD Bullish Cross: %s\n', ...
            currentRSI, iff(isOversold, 'YES', 'NO'), iff(isMacdBullishCross, 'YES', 'NO'));
        
        % Check for absolute technical convergence parameters before allowing capital deployment
        if isOversold || isMacdBullishCross
            % (was: <= 21.50 / <= 22.00 / <= 22.80 fixed-dollar tiers)
            if pctOfTrailingRange <= 0.05
                gridAccumulationPct = 1.00; 
                currentTierLabel = "Tier 4 (Maximum Margin-of-Safety Cap reached)";
            elseif pctOfTrailingRange <= 0.15
                gridAccumulationPct = 0.60; 
                currentTierLabel = "Tier 3 (Structural Support Load Tier)";
            elseif pctOfTrailingRange <= 0.25
                gridAccumulationPct = 0.30; 
                currentTierLabel = "Tier 2 (Incremental Entry Multi-Unit Target)";
            else
                gridAccumulationPct = 0.10; 
                currentTierLabel = "Tier 1 (Initial Strategic Value Entry)";
            end
            
            % If a bullish momentum cross validates an oversold floor, amplify allocation layout by 1.2x
            if isOversold && isMacdBullishCross
                gridAccumulationPct = min(1.00, gridAccumulationPct * 1.20);
                currentTierLabel = strcat(currentTierLabel, " + CONFLUENCE BOOST");
            end
            
            targetCapitalDeployment = allocatedTargetBankroll * gridAccumulationPct;
            sharesToAccumulate = floor(targetCapitalDeployment / currentMarketPrice);
            systemVote = sprintf('🎣 VALUE ACCUMULATION: %s', currentTierLabel);
            finalAllocation = (targetCapitalDeployment / allocatedTargetBankroll) * 100;
            noBuyLine = sprintf('✅ Bottom-Fishing Order Triggered: BUY %d SHARES at market\n', sharesToAccumulate);
        else
            % Price is in the grid zone, but technical indicators warn that the floor hasn't formed yet
            sharesToAccumulate = 0;
            systemVote = "⏸️ GRID PASSIVE HOLD (Awaiting RSI Oversold or MACD Bullish Pivot)";
            finalAllocation = 0.00;
            noBuyLine = sprintf('🚫 Strategic Asset Action: AVOID BUY TODAY (No technical turn signal)\n');
        end
    else
        % Keep standard BiLSTM production metrics if price remains above value target zones
        sharesToAccumulate = 0;
        frame45Trend = final45FrameReturns(end); 
        
        if frame45Trend > 0.015 && currentLLMSentiment > 0.1
            systemVote = "🚀 MULTIMODAL CONFLUENCE BUY (EKF Smooth Price + LLM Positive)";
            finalAllocation = 20.00;
            noBuyLine = '';
        elseif frame45Trend < -0.015
            systemVote = "⚠️ CONFLUENCE RISK EXIT (Bearish Trend Target Detected)";
            finalAllocation = 0.00;
            noBuyLine = sprintf('🚫 Strategic Asset Action: AVOID BUY TODAY\n');
        else
            systemVote = "⏸️ NEUTRAL HOLD";
            finalAllocation = 0.00;
            noBuyLine = sprintf('🚫 Strategic Asset Action: AVOID BUY TODAY\n');
        end
    end

    rawPred15 = final45FrameReturns(min(length(final45FrameReturns), 15)); 
    rawPred45 = final45FrameReturns(end);                                 
    finalCapitalAllocationPct = finalAllocation;
    % -------------------------------------------------------------------
    %  ENHANCEMENT: DYNAMIC FORWARD HORIZON HORIZON PIVOT SCANNER
    % -------------------------------------------------------------------
    % Find the absolute lowest projected price point in the 45-day path
    [minProjectedPrice, minIdx] = min(predictedPricePath);
    pivotDate = futureDates(minIdx);
    
    % Define an accumulation buying window around that terminal floor (1 business day before/after)
    windowStart = futureDates(max(1, minIdx - 1));
    windowEnd = futureDates(min(horizonAhead, minIdx + 1));
    
    % Generate dynamic forward advice text strings for the display engine
    if minIdx > 1 && (basePrice - minProjectedPrice) / basePrice > 0.05
        % System detects a major structural dip before a turnaround rally
        forwardAdviceLine = sprintf('⏳ Cycle Strategy: AVOID entry until floor stabilizes around %s.', datestr(windowStart, 'mmm dd'));
        forwardActionLine = sprintf('🎣 Accumulation Target: Build long layers between %s and %s.', datestr(windowStart, 'mmm dd'), datestr(windowEnd, 'mmm dd'));
        targetFloorLine  = sprintf('🎯 Projected Cycle Floor Price: $%.2f', minProjectedPrice);
    else
        % System detects a steady upward trajectory or flat matrix
        forwardAdviceLine = sprintf('🚀 Cycle Strategy: Bullish structure intact. Immediate accumulation viable.');
        forwardActionLine = sprintf('💼 Allocation Plan: Maintain standard core portfolio sizing scales.');
        targetFloorLine  = sprintf('🎯 Projected Cycle Floor Price: $%.2f', basePrice);
    end

       %% ===================================================================
    %         PART 7 — EXECUTIVE FORWARD FORECAST DASHBOARD (UI TABLE)
    %% ===================================================================
    figName = sprintf('%s rev5 Multi-Modal Forward 45-Day Horizon Forecast', symbol);
    
    % FIX: Expand figure width slightly to provide comfortable margins for a strategy table
    figure('Name', figName, 'Color', [1 1 1], 'Position', [100, 100, 1050, 650]);

    % -------------------------------------------------------------------
    % SUBPLOT 1: Unified Forecast Core Axis (Takes up the upper half)
    % -------------------------------------------------------------------
    subplot(2,1,1);
    hold on;

    plot(futureDates, predictedPricePath, '--b', 'LineWidth', 1.8, ...
         'Marker', 'o', 'MarkerFaceColor', 'b', 'DisplayName', 'LSTM Vector Path');

    % ---------------------------------------------------------------
    % ENHANCEMENT: Resistance (awareness) / Support (stop-loss gating)
    % overlay lines, computed above from swing-pivot clustering.
    %
    % FIX: labels were previously split between 'left' (resistance/
    % support) and 'right' (stop-loss) alignment. 'left' collided with
    % the LSTM path's own early markers whenever the forecast opened
    % near one of these levels (seen on the INTU run), and 'right' had
    % zero margin against the axis edge so the stop-loss label got
    % clipped in the exported figure (seen on both INTU and FISV runs).
    % All three now render 'right', landing in the blank padding zone
    % added to xlim below instead of over data or off the edge.
    % ---------------------------------------------------------------
    yline(resistanceLevel, '--', sprintf('Resistance $%.2f', resistanceLevel), ...
        'Color', [0.80 0.10 0.10], 'LineWidth', 1.4, 'FontWeight', 'bold', ...
        'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', ...
        'DisplayName', sprintf('Resistance $%.2f (%d touches)', resistanceLevel, resistanceStrength));

    yline(supportLevel, '--', sprintf('Support $%.2f', supportLevel), ...
        'Color', [0.10 0.55 0.15], 'LineWidth', 1.4, 'FontWeight', 'bold', ...
        'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'top', ...
        'DisplayName', sprintf('Support $%.2f (%d touches)', supportLevel, supportStrength));

    yline(suggestedStopLoss, ':', sprintf('Stop-Loss $%.2f', suggestedStopLoss), ...
        'Color', [0.55 0.55 0.55], 'LineWidth', 1.1, ...
        'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', ...
        'DisplayName', sprintf('Stop-Loss Trigger $%.2f', suggestedStopLoss));

    grid on; 
    displayStartTime = futureDates(1);
    displayEndTime = futureDates(horizonAhead);
    labelPadding = caldays(3);  % blank breathing room so right-aligned labels never clip or overlap data
    xlim([displayStartTime, displayEndTime + labelPadding]);

    % Widen y-limits so resistance/support/stop lines are never clipped
    % off the top/bottom of the axis when they sit outside the
    % forecast path's own min/max.
    pathMin = min(predictedPricePath); pathMax = max(predictedPricePath);
    yLowBound  = min([pathMin, supportLevel, suggestedStopLoss]) * 0.985;
    yHighBound = max([pathMax, resistanceLevel]) * 1.015;
    ylim([yLowBound, yHighBound]);

    title(sprintf('%s Multi-Modal %d‑Frame Consolidated Vector Forecast', symbol, horizonAhead));
    legend('Location', 'best'); 
    ylabel('Projected Price ($)');
    hold off;

    % -------------------------------------------------------------------
    % SUBPLOT 2 ALTERNATIVE: Actionable Executive Strategy Table Widget
    % -------------------------------------------------------------------
    % Parse text parameters safely to prevent character wrapping errors
    endVectorReturnPct = ((predictedPricePath(end) - basePrice) / basePrice) * 100;
    
    % Construct the data rows for the user interface table
    tableData = {
        'Expected Return @ Frame-45', sprintf('%+.2f%%', endVectorReturnPct);
        'Target Vector Price Target', sprintf('$%.2f', predictedPricePath(end));
        'System Action Status',       char(systemVote);
        'Active Current LLM Sentiment', sprintf('%+.2f', currentLLMSentiment);
        'Sentiment Data Source',        char(sentimentSourceLabel);
        'Optimal Sizing Allocation',  sprintf('%.2f%% of Capital', finalAllocation);
        'Operational Cycle Strategy',  strtrim(forwardAdviceLine);
        'Accumulation Target Window', strtrim(forwardActionLine);
        'Projected Cycle Floor Price', strtrim(targetFloorLine);
        'Computed Resistance (Awareness)', sprintf('$%.2f  (%d touches, trailing %dd)', resistanceLevel, resistanceStrength, lookbackRange);
        'Computed Support (Stop-Loss Gate)', sprintf('$%.2f  (%d touches, trailing %dd)', supportLevel, supportStrength, lookbackRange);
        'Suggested Stop-Loss Trigger', sprintf('$%.2f  (Support - 0.5x ATR buffer)', suggestedStopLoss)
    };

    % Establish clean column identifiers
    columnNames = {'Strategic Metric Parameter', 'System Production Advisory / Value Output'};

    % Instantiate the native interactive table widget on the lower half of the figure window
    % Position format: [left, bottom, width, height] relative to figure size
    tbl = uitable('Data', tableData, ...
                  'ColumnName', columnNames, ...
                  'RowName', [], ... % Disables index numbers on rows to save space
                  'Units', 'normalized', ...
                  'Position', [0.05, 0.05, 0.90, 0.40], ...
                  'FontSize', 10, ...
                  'FontWeight', 'bold');

    % Auto-scale column widths so long text sentences are fully visible without truncation
    tbl.ColumnWidth = {220, 700};

    drawnow; commandwindow;
    fprintf('🎉 rev5 Multi-Step Engine Table Cycle Execution Complete.\n');
end


function val = iff(condition, trueVal, falseVal)
    if condition
        val = trueVal; 
    else
        val = falseVal; 
    end
end

function refreshCacheIfStale(csvPath, dateColumnName, staleDays)
% REFRESHCACHEIFSTALE  Delete csvPath if it's missing/corrupt/too-short,
% OR if its newest dated row is more than `staleDays` calendar days
% older than today. A caller that follows this with
% `if ~exist(csvPath,'file') ... download ... end` transparently
% refreshes stale local Tiingo caches instead of silently reusing old
% market data forever (the original bug: caches were only ever
% invalidated for outright corruption, never for age).
    if ~exist(csvPath, 'file')
        return;  % nothing to invalidate -- caller's download path handles this
    end
    try
        cacheTable = readtable(csvPath, 'FileType', 'text', 'VariableNamingRule', 'preserve');
        if isempty(cacheTable) || height(cacheTable) < 5
            error('Empty or too-short cache file');
        end
        varNames = cacheTable.Properties.VariableNames;
        dateIdx = find(strcmpi(varNames, dateColumnName), 1);
        if isempty(dateIdx)
            error('Missing expected date column ''%s''', dateColumnName);
        end
        cacheDates = datetime(cacheTable.(varNames{dateIdx}), 'InputFormat', 'yyyy-MM-dd');
        cacheAgeDays = days(datetime('today') - max(cacheDates));
        if cacheAgeDays > staleDays
            fprintf("🕒 Local cache '%s' is %.0f day(s) old (newest row %s). Refreshing from Tiingo...\n", ...
                csvPath, cacheAgeDays, datestr(max(cacheDates), 'yyyy-mm-dd'));
            delete(csvPath);
        end
    catch
        fprintf("⚠️ Corrupt or unreadable local cache '%s'. Force-clearing...\n", csvPath);
        if exist(csvPath, 'file'), delete(csvPath); end
    end
end

function [resistanceLevel, supportLevel, resistanceStrength, supportStrength] = ...
    computeSupportResistance(h_High, h_Low, currentPrice, recentATR, lookbackRange)
% COMPUTESUPPORTRESISTANCE  Fractal swing-pivot support/resistance finder.
%
% Detects swing highs (in h_High) and swing lows (in h_Low) over the
% trailing `lookbackRange` bars using a classic +/-3-bar fractal rule (a
% bar is a swing high/low only if it is the STRICT extreme of its own
% 7-bar neighborhood, i.e. 3 bars each side). Nearby pivots are then
% merged into single "levels" using an ATR-scaled tolerance, and the
% level's strength is its touch count (how many pivots clustered into
% it -- a simple stand-in for level confluence).
%
% Returns the nearest clustered level ABOVE currentPrice as resistance
% and the nearest clustered level BELOW currentPrice as support. Falls
% back to the raw trailing high/low if no qualifying pivot exists on
% that side (e.g., price sitting at a fresh 1yr high or low).

    N = numel(h_High);
    fractalArm = 3;
    idxStart = max(1, N - lookbackRange + 1);

    hi = h_High(idxStart:N);
    lo = h_Low(idxStart:N);
    numBars = numel(hi);

    isSwingHigh = false(numBars, 1);
    isSwingLow  = false(numBars, 1);

    for k = (fractalArm + 1):(numBars - fractalArm)
        windowHi = hi(k - fractalArm : k + fractalArm);
        windowLo = lo(k - fractalArm : k + fractalArm);
        if hi(k) == max(windowHi) && sum(windowHi == hi(k)) == 1
            isSwingHigh(k) = true;
        end
        if lo(k) == min(windowLo) && sum(windowLo == lo(k)) == 1
            isSwingLow(k) = true;
        end
    end

    swingHighPrices = hi(isSwingHigh);
    swingLowPrices  = lo(isSwingLow);

    % Cluster tolerance: half an ATR, with a small floor so it never
    % collapses to zero for extremely quiet/low-priced names.
    clusterTol = max(0.5 * recentATR, 1e-3 * max(currentPrice, 1));

    [resistanceLevel, resistanceStrength] = ...
        nearestClusteredLevel(swingHighPrices, currentPrice, clusterTol, 'above');
    [supportLevel, supportStrength] = ...
        nearestClusteredLevel(swingLowPrices, currentPrice, clusterTol, 'below');

    % Fallback: no qualifying pivot on that side (price at/near the
    % trailing extreme) -- use the raw trailing high/low instead so the
    % caller always gets a usable level.
    if isnan(resistanceLevel)
        resistanceLevel = max(hi);
        resistanceStrength = 1;
    end
    if isnan(supportLevel)
        supportLevel = min(lo);
        supportStrength = 1;
    end
end

function [level, strength] = nearestClusteredLevel(pivotPrices, currentPrice, tol, side)
% NEARESTCLUSTEREDLEVEL  Pick the nearest qualifying pivot to currentPrice
% on the requested side ('above' or 'below'), then merge every other
% pivot within `tol` dollars of it into the same level (averaged) and
% report the merged count as the level's touch-count strength.
    level = NaN;
    strength = 0;
    if isempty(pivotPrices)
        return;
    end

    switch side
        case 'above'
            candidates = pivotPrices(pivotPrices > currentPrice);
        case 'below'
            candidates = pivotPrices(pivotPrices < currentPrice);
        otherwise
            error('nearestClusteredLevel: side must be ''above'' or ''below''.');
    end
    if isempty(candidates)
        return;
    end

    candidates = sort(candidates);
    if strcmp(side, 'above')
        seedPrice = candidates(1);     % closest pivot above price
    else
        seedPrice = candidates(end);   % closest pivot below price
    end

    inCluster = abs(candidates - seedPrice) <= tol;
    level = mean(candidates(inCluster));
    strength = sum(inCluster);
end

function newsItems = fetchTiingoNews(symbol, apiKey, limitCount)
% Fetch recent news headlines/descriptions for a ticker from Tiingo's News
% API (https://api.tiingo.com/tiingo/news). Returns a struct array with
% (at minimum) .title and .publishedDate fields per Tiingo's documented
% news schema.
    newsUrl = sprintf(['https://api.tiingo.com/tiingo/news?tickers=%s&' ...
        'limit=%d&sortBy=publishedDate&token=%s'], lower(symbol), limitCount, apiKey);
    opts = weboptions('ContentType', 'json', 'Timeout', 30);
    newsItems = webread(newsUrl, opts);
    if isempty(newsItems)
        error('No news items returned for %s -- check ticker coverage / news entitlement on your Tiingo plan.', symbol);
    end
end

function llmTable = scoreNewsSentimentWithClaude(newsItems, claudeApiKey)
% Ask Claude to score each headline's sentiment (-1..+1) in a single call,
% then average per calendar date to match the llm_sentiment_score cache
% schema the rest of the pipeline already expects (date, score columns).
    numItems = numel(newsItems);
    linesForPrompt = strings(numItems, 1);
    itemDates = strings(numItems, 1);

    for i = 1:numItems
        item = newsItems(i);
        % Tiingo publishedDate is ISO-8601, e.g. '2026-09-24T14:05:00.000Z'.
        % Fall back to a plain date-only parse if the timestamp portion
        % doesn't match, rather than erroring out the whole batch.
        try
            itemDT = datetime(item.publishedDate, 'InputFormat', "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", 'TimeZone', 'UTC');
        catch
            itemDT = datetime(item.publishedDate(1:10), 'InputFormat', 'yyyy-MM-dd');
        end
        itemDates(i) = string(itemDT, 'yyyy-MM-dd');
        linesForPrompt(i) = sprintf('%d. [%s] %s', i, itemDates(i), string(item.title));
    end

    promptText = sprintf([ ...
        "You are a financial news sentiment scorer. For each numbered headline below, " ...
        "assign a sentiment score from -1.0 (very bearish for the stock) to +1.0 (very bullish), " ...
        "with 0.0 being neutral. Respond with ONLY a JSON array, no other text, no markdown " ...
        "fences, in the form [{""i"":1,""score"":0.3}, ...] with exactly one object per " ...
        "headline, in the same order given.\n\n%s"], strjoin(linesForPrompt, newline));

    bodyStruct = struct( ...
        'model', 'claude-sonnet-4-6', ...
        'max_tokens', 2048, ...
        'messages', {{struct('role', 'user', 'content', char(promptText))}});

    opts = weboptions( ...
        'HeaderFields', {'x-api-key', char(claudeApiKey); 'anthropic-version', '2023-06-01'}, ...
        'MediaType', 'application/json', ...
        'Timeout', 60, ...
        'ContentType', 'json');

    response = webwrite('https://api.anthropic.com/v1/messages', bodyStruct, opts);
    rawText = response.content(1).text;
    cleanText = regexprep(rawText, '```json|```', '');
    scores = jsondecode(strtrim(cleanText));

    perHeadlineScore = nan(numItems, 1);
    for k = 1:numel(scores)
        perHeadlineScore(scores(k).i) = scores(k).score;
    end

    [uniqueDates, ~, dateIdx] = unique(itemDates);
    avgScorePerDate = accumarray(dateIdx, perHeadlineScore, [], @(v) mean(v, 'omitnan'));

    llmTable = table(uniqueDates(:), avgScorePerDate(:), ...
        'VariableNames', {'date', 'llm_sentiment_score'});
end
