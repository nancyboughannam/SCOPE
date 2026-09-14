function [] = plotGenericLine(rowOfset, columnOfset, yLabel, appType, legendPos, calculatePercentage, divisor, ignoreZeroValues, hideLowerValues, hideXAxis)
    folderPath = getConfiguration(1);
    numOfSimulations = getConfiguration(3);
    stepOfxAxis = getConfiguration(4);
    scenarioType = getConfiguration(5);

    % ===== DISPLAY POLICY NAMES CHANGE START =====
    displayPolicyNames = getConfiguration(6);

    if numel(displayPolicyNames) ~= numel(scenarioType)
     error('Every raw policy identifier must have one display name.');
    end
% ===== DISPLAY POLICY NAMES CHANGE END =====


    startOfMobileDeviceLoop = getConfiguration(10);
    stepOfMobileDeviceLoop = getConfiguration(11);
    endOfMobileDeviceLoop = getConfiguration(12);
    numOfMobileDevices = floor((endOfMobileDeviceLoop - startOfMobileDeviceLoop)/stepOfMobileDeviceLoop) + 1;

    all_results = zeros(numOfSimulations, size(scenarioType,2), numOfMobileDevices);
    min_results = zeros(size(scenarioType,2), numOfMobileDevices);
    max_results = zeros(size(scenarioType,2), numOfMobileDevices);
    
    if ~exist('appType','var')
        appType = 'ALL_APPS';
    end
    
    if ~exist('divisor','var')
        divisor = 1;
    end
    
    if ~exist('ignoreZeroValues','var')
        ignoreZeroValues = 0;
    end
    
    if ~exist('hideLowerValues','var')
        hideLowerValues = 0;
    end
    
    if exist('hideXAxis','var')
        hideXAxisStartValue = hideXAxis(2);
        hideXAxisIndex = hideXAxis(1); 
    end

    for s=1:numOfSimulations
        for i=1:size(scenarioType,2)
            for j=1:numOfMobileDevices
        try
            mobileDeviceNumber = startOfMobileDeviceLoop + stepOfMobileDeviceLoop * (j-1);
            
            % 1. Build the filename
            fileName = strcat('SIMRESULT_ITS_SCENARIO_', char(scenarioType(i)), '_', ...
                              int2str(mobileDeviceNumber), 'DEVICES_', appType, '_GENERIC.log');
            
    % ===== ITERATION FOLDER FIX CHANGE START =====
             iterationFolder = sprintf('ite%d', s);
             filePath = fullfile(folderPath, iterationFolder, fileName);
    % ===== ITERATION FOLDER FIX CHANGE END =====
            % 3. Check if file exists before reading
            if exist(filePath, 'file')
                readData = dlmread(filePath, ';', rowOfset, 0);
                value = readData(1, columnOfset);
                
                if(calculatePercentage == 1)
                    readDataHeader = dlmread(filePath, ';', 1, 0);
                    totalTask = readDataHeader(1, 1) + readDataHeader(1, 2);
                    value = (100 * value) / totalTask;
                end
                all_results(s, i, j) = value;
            else
                % If file is missing (like 100DEVICES), just skip it
                fprintf('Skipping missing file: %s\n', filePath);
                all_results(s, i, j) = NaN; 
            end
        catch err
            fprintf('Error processing %s: %s\n', fileName, err.message);
        end
    end
        end
    end
        
%     if(numOfSimulations == 1)
%         results = all_results;
%     else
%         if(ignoreZeroValues == 1)
%             results = sum(all_results,1) ./ sum(all_results~=0,1);
%             %TODO cahnge NaN to 0
%         else
%             results = mean(all_results); %still 3d matrix but 1xMxN format
%         end
%     end
% 
% results = squeeze(results);
% 
%     for i=1:size(scenarioType,2)
%         for j=1:numOfMobileDevices
%             if(results(i,j) < hideLowerValues)
%                 results(i,j) = NaN;
%             else
%                 results(i,j) = results(i,j) / divisor;
%             end                
%         end
%     end
% 
%     if exist('hideXAxis','var')
%         for j=1:numOfMobileDevices
%             if(j*stepOfMobileDeviceLoop+startOfMobileDeviceLoop > hideXAxisStartValue)
%                 results(hideXAxisIndex,j) = NaN;
%             end
%         end
%     end
% 
%     for i=1:size(scenarioType,2)
%         for j=1:numOfMobileDevices
%             x=results(i,j);                    % Create Data
%             SEM = std(x)/sqrt(length(x));            % Standard Error
%             ts = tinv([0.05  0.95],length(x)-1);   % T-Score
%             CI = mean(x) + ts*SEM;                   % Confidence Intervals
% 
%             if(CI(1) < 0)
%                 CI(1) = 0;
%             end
% 
%             if(CI(2) < 0)
%                 CI(2) = 0;
%             end
% 
%             min_results(i,j) = results(i,j) - CI(1);
%             max_results(i,j) = CI(2) - results(i,j);
%         end
%     end
% 
%     types = zeros(1,numOfMobileDevices);


% ===== MULTI-ITERATION STATISTICS CHANGE START =====
results = NaN(size(scenarioType,2), numOfMobileDevices);

for i = 1:size(scenarioType,2)
    for j = 1:numOfMobileDevices

        % Obtain this policy/vehicle value from every iteration
        rawX = squeeze(all_results(:,i,j));

        % Verify that all configured iterations were found
        availableCount = sum(~isnan(rawX));

        if availableCount ~= numOfSimulations
            vehicleCount = startOfMobileDeviceLoop + ...
                ((j-1) * stepOfMobileDeviceLoop);

            error(['Incomplete iterations for policy %s at %d vehicles: ' ...
                   'expected %d but found %d.'], ...
                   char(scenarioType(i)), ...
                   vehicleCount, ...
                   numOfSimulations, ...
                   availableCount);
        end

        x = rawX;

        if ignoreZeroValues == 1
            x = x(x ~= 0);
        end

        x = x(~isnan(x));
        x = x ./ divisor;
        sampleCount = numel(x);

        if sampleCount == 0
            continue;
        end

        meanValue = mean(x);

        if meanValue < hideLowerValues
            continue;
        end

        results(i,j) = meanValue;

        if sampleCount == 1
            min_results(i,j) = 0;
            max_results(i,j) = 0;
        else
            SEM = std(x,0) / sqrt(sampleCount);
            tCritical = tinv(0.975, sampleCount - 1);
            halfWidth = tCritical * SEM;

            lowerBound = max(0, meanValue - halfWidth);
            upperBound = meanValue + halfWidth;

            min_results(i,j) = meanValue - lowerBound;
            max_results(i,j) = upperBound - meanValue;
        end
    end
end

if exist('hideXAxis','var')
    for j = 1:numOfMobileDevices
        vehicleCount = startOfMobileDeviceLoop + ...
            ((j-1) * stepOfMobileDeviceLoop);

        if vehicleCount > hideXAxisStartValue
            results(hideXAxisIndex,j) = NaN;
            min_results(hideXAxisIndex,j) = NaN;
            max_results(hideXAxisIndex,j) = NaN;
        end
    end
end
% ===== MULTI-ITERATION STATISTICS CHANGE END =====
    for i=1:numOfMobileDevices
        types(i)=startOfMobileDeviceLoop+((i-1)*stepOfMobileDeviceLoop);
    end
    
    hFig = figure;
    pos=getConfiguration(7);
    fontSizeArray = getConfiguration(8);
    set(hFig, 'Units','centimeters');
    set(hFig, 'Position',pos);
    set(0,'DefaultAxesFontName','Times New Roman');
    set(0,'DefaultTextFontName','Times New Roman');
    set(0,'DefaultAxesFontSize',fontSizeArray(3));

  % --- CUSTOM COLOR LOGIC (THE "HERO" STYLE) ---
    for j=1:size(scenarioType,2)
        policyName = char(scenarioType(j));
        
        % Default Styles
        pColor = [0.2 0.2 0.2];
        pMarker = 'o';
        pLineStyle = '-';
        pLineWidth = 1.0;
        pFaceColor = 'none';
        pMarkerSize = 6;
        
        % Assign Specific Styles based on Policy Name
        if contains(policyName, 'RT_OPT_ML')
            % The Hero: Bold Blue, Solid Circles
            pColor = [0 0.4470 0.7410]; 
            pMarker = 'o';
            pLineStyle = '-';
            pLineWidth = 2.5; 
            pFaceColor = [0 0.4470 0.7410];
            pMarkerSize = 8;
        elseif strcmp(policyName, 'EXP3_MC')
            % The Rival: Orange, Triangles
            pColor = [0.8500 0.3250 0.0980];
            pMarker = '^';
            pLineStyle = '-';
            pLineWidth = 1.5;
            pFaceColor = 'none';
        elseif contains(policyName, 'RAIDER')
            % Yellow/Gold, Diamonds
            pColor = [0.9290 0.6940 0.1250];
            pMarker = 'd';
            pLineStyle = '--';
            pLineWidth = 1.5;
            pFaceColor = 'none';
        elseif contains(policyName, 'RANDOM')
            % Grey, Squares
            pColor = [0.5 0.5 0.5];
            pMarker = 's';
            pLineStyle = ':';
            pLineWidth = 1.5;
            pFaceColor = 'none';
        else
            % Fallback: Use config colors if unknown
            pColor = getConfiguration(20+j);
        end
        
        % Plotting
        if(getConfiguration(19) == 1)
            errorbar(types, results(j,:), min_results(j,:), max_results(j,:), ...
                'Color', pColor, ...
                'Marker', pMarker, ...
                'LineStyle', pLineStyle, ...
                'LineWidth', pLineWidth, ...
                'MarkerFaceColor', pFaceColor, ...
                'MarkerSize', pMarkerSize);
        else
            plot(types, results(j,:), ...
                'Color', pColor, ...
                'Marker', pMarker, ...
                'LineStyle', pLineStyle, ...
                'LineWidth', pLineWidth, ...
                'MarkerFaceColor', pFaceColor, ...
                'MarkerSize', pMarkerSize);
        end
        hold on;
    end
    % ---------------------------------------------


      set(gca,'color','none');
 
     % ===== GRAPH LEGEND DISPLAY NAMES CHANGE START =====
    legends = displayPolicyNames;
    lgnd = legend(legends, 'Location', legendPos);
    % ===== GRAPH LEGEND DISPLAY NAMES CHANGE END =====

    if(getConfiguration(20) == 1)
        set(lgnd,'color','none');
    end
    
    xCoefficent = 100;
    if(getConfiguration(17) == 0)
        xCoefficent = 1;
    end
    
    hold off;
    axis square
    xlabel(getConfiguration(9));
    step = stepOfxAxis*stepOfMobileDeviceLoop;
    set(gca,'XTick', step:step:endOfMobileDeviceLoop);
    set(gca,'XTickLabel', step/xCoefficent:step/xCoefficent:endOfMobileDeviceLoop/xCoefficent);
    ylabel(yLabel);
    set(gca,'XLim',[startOfMobileDeviceLoop-5 endOfMobileDeviceLoop+5]);
    
    if(getConfiguration(17) == 1)
        xlim = get(gca,'XLim');
        ylim = get(gca,'YLim');
        text(1.02 * xlim(2), 0.165 * ylim(2), 'x 10^2');
    end
    
    
    set(get(gca,'Xlabel'),'FontSize',fontSizeArray(1));
    set(get(gca,'Ylabel'),'FontSize',fontSizeArray(1));
    set(lgnd,'FontSize',fontSizeArray(2));
    
    if(getConfiguration(18) == 1)
        set(hFig, 'PaperUnits', 'centimeters');
        set(hFig, 'PaperPositionMode', 'manual');
        set(hFig, 'PaperPosition',[0 0 pos(3) pos(4)]);
        set(gcf, 'PaperSize', [pos(3) pos(4)]); %Keep the same paper size
        filename = strcat(folderPath,'/',int2str(rowOfset),'_',int2str(columnOfset),'_',appType);
        exportgraphics(gcf, [filename '.pdf'], 'ContentType', 'vector');
    end
end