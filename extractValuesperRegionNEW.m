folderPath = '/Volumes/Projects/Student_projects/20_Carolin_Hambrock_2023/output/T2 registraition/T2w/region T1/parental'; % Update this with the path to your folder
files = dir(fullfile(folderPath, '*.csv'));
inputFiles = {files.name};

targetRegions = [629, 685, 718, 725, 733, 741, 2629, 2685,2718,2725,2733,2741];  % Add more region numbers as needed

tables = extractValues(inputFiles, targetRegions, folderPath);

function [individualTables, overallTable] = extractValues(inputFiles, targetRegions, folderPath)
    individualTables = cell(1, length(inputFiles));
    ratioTables = cell(1, length(inputFiles)); % Store individual ratio tables

    for fileIndex = 1:length(inputFiles)
        inputFile = fullfile(folderPath, inputFiles{fileIndex});

        % Read the input file
        tableData = readtable(inputFile);

        % Initialize results containers for the current file
        resultsT2 = containers.Map('KeyType', 'double', 'ValueType', 'double');
        resultsSize = containers.Map('KeyType', 'double', 'ValueType', 'double');

        % Process each row in the table
        for rowIndex = 1:height(tableData)
            regionNumber = tableData.ARAID(rowIndex);
            if ismember(regionNumber, targetRegions)
                % Extract the T2 value
                valueT2 = tableData.T2Value(rowIndex);
                resultsT2(regionNumber) = valueT2;
                
                % Extract the size value
                valueSize = tableData.RegionSize(rowIndex);
                resultsSize(regionNumber) = valueSize;
            end
        end

        % Create a table with all target regions and NaN values
        tableData = table(targetRegions', NaN(length(targetRegions),1), NaN(length(targetRegions), 1), NaN(length(targetRegions), 1), ...
            'VariableNames', {'Region', 'ValuesT2', 'ValuesSize', 'Ratio'});

        % Update the values for the existing regions
        existingRegionsT2 = cell2mat(resultsT2.keys);
        existingValuesT2 = cell2mat(resultsT2.values);
        [~, locT2] = ismember(existingRegionsT2, targetRegions);
        tableData.ValuesT2(locT2) = existingValuesT2;

        existingRegionsSize = cell2mat(resultsSize.keys);
        existingValuesSize = cell2mat(resultsSize.values);
        [~, locSize] = ismember(existingRegionsSize, targetRegions);
        tableData.ValuesSize(locSize) = existingValuesSize;

        % Calculate the ratio (ValuesT2 / ValuesSize) and update the table
        ratio = existingValuesT2 ./ existingValuesSize;
        tableData.Ratio(locT2) = ratio;

        % Store the individual ratio table
        ratioTables{fileIndex} = tableData;
        
        % Store the table in the workspace with the original file name as the variable name
        [~, fileName, ~] = fileparts(inputFiles{fileIndex});
        variableName = matlab.lang.makeValidName(fileName);
        assignin('base', variableName, tableData);
        individualTables{fileIndex} = tableData;
    end

    % Create an overall table with all target regions and ratio values for each input file
    overallTable = table(targetRegions');
    for fileIndex = 1:length(inputFiles)
        [~, fileName, ~] = fileparts(inputFiles{fileIndex});
        variableName = matlab.lang.makeValidName(fileName);
        overallTable.(variableName) = ratioTables{fileIndex}.Ratio;
    end
    
    % Display the overall table in the MATLAB workspace
    assignin('base', 'allRatios', overallTable);
end
