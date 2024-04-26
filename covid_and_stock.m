%% Machine learning based prediction of stock prices under pandemic influence
%  Fitrensemble(bagged decision trees) is used to predict future stock prices under pendemic
%  influences. Data from the past are compounded into samples to train and
%  test the model; a seperate ensemble is responsible for every output group.
%
% Two structures are tested(comp). Later layers:
%   True) use predictions produced by earlier layers as extra input
%   False) they do not. Each layer acts independently from other layers. 

clear;
clc;

%% Settings
% A week of stock data is used minimum regardeless of the "groupings".
% Since stock data have gaps in them, every "day" is defined over only the
% avalible days.
%
% Feature list:
% Ticker, section, days,
% [open high low adjClose volume] of current day + 6 days in the past,
% average adjClose within a group, for total_groups into the past,
%
% [n_cases n_deaths] of current + 6 days in the past,
% [avnc avnd] of days_gap in the past, for total_gaps
%
% and more if avalible
%
% Label: average adjClose of the stock within days into the future.
%
% Back_protect: these are removed from the train/valid set to simulate using
% the model on newly acquired data.

% Sample range
group_in_days_of = 10; % 10 is half a month
total_groups = 10;     % 5 months in the past total
covid_days_gap = 7;    % 7 days of covid average
covid_total_gaps = 8;  % 2 months of covid in total

% Label range
label_days = 10;       % predict the average of 10 business days -> 2 weeks
label_groups = 18;     % make a total of 18 predictions -> 9 months total

% Train/Validation/Test
back_protect = 0.25;   % what % of everything is used to test
train_per = 0.8;       % what %(of what is left) is used to train(vs valid)

% model structure: later layers do(T)/DO NOT(F) use prediction from
%                  privious layers as extra input
comp = false;

% discard_trained_structures:
%       ignore existing saved structures like sample/label matrix and
%       trained trees
% If false, remember to delete all saved structure files after
% changing any parameter
discard_trained_structures = false;

%% Import: covid\us_epidemiology.csv
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Date", "Cases", "Deaths", "New_cases", "New_deaths", "Avg_new_cases", "Avg_new_deaths"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");
try
    usepidemiology = readtable("covid\us_epidemiology.csv", opts);
catch Ex
    usepidemiology = readtable("covid/us_epidemiology.csv", opts);
end
clear opts

%% Import: stocks\us_stock.csv
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume", "Section", "stock_ticker"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double", "categorical", "categorical"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, ["Section", "stock_ticker"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");
try
    usstock = readtable("stocks\us_stock.csv", opts);
catch Ex
    usstock = readtable("stocks/us_stock.csv", opts);
end
clear opts

%% Convert dates to "days from 2000-01-01" (Y/M/D) -> mostly 7-8k
%  and tickers to id's. Everything have to be numbers.
temp = table(daysact(datetime(2000, 01, 01), usepidemiology.Date));
temp = renamevars(temp, {'Var1'}, 'Days');
covid_raw = [temp removevars(usepidemiology, {'Date'})];
clear temp;

temp = table(daysact(datetime(2000, 01, 01), usstock.Date));
temp = renamevars(temp, {'Var1'}, 'Days');
stock_raw = [temp removevars(removevars(usstock, {'Date'}), {'Close'})];
clear temp;

temp_section = table(grp2idx(categorical(usstock.Section)));
temp_section = renamevars(temp_section, {'Var1'}, 'Section');
temp_ticker = table(grp2idx(categorical(usstock.stock_ticker)));
temp_ticker = renamevars(temp_ticker, {'Var1'}, 'Stock_ticker');
stock_raw = [removevars(removevars(stock_raw, {'Section'}), {'stock_ticker'}) temp_section temp_ticker];
clear temp_section temp_ticker;

%% Construct and reuse the covid suffixes
%  These are attached to the end of every sample entry of the coresponding
%  date. Precalculated and reused to save time. 
covid_suffix = [];
day_start = min(covid_raw.Days)+8 ...
    +covid_days_gap*covid_total_gaps;
day_end = max(covid_raw.Days);

if ~isfile("covid_suffix.mat") || discard_trained_structures 
    fprintf("Constructing covid_suffix...\n");
    for current_day_index = 1:1:size(covid_raw.Days, 1)
        if (covid_raw.Days(current_day_index)<day_start)
            continue;
        elseif (covid_raw.Days(current_day_index)>day_end)
            break;
        end
    
        if (mod(current_day_index, 100) == 0)
            fprintf("Day: %d of %d...\n", current_day_index, size(covid_raw.Days, 1));
        end
        each_day = [covid_raw.Days(current_day_index)];
        % for every valid day, include the 0-6 past days
        for into_past = 0:1:6
            pd_covid = covid_raw(current_day_index-into_past, :);
            each_day = [each_day table2array(pd_covid(1, 4:5))];
        end
        % for group, get avnc and avnd
        for group = 0:1:covid_total_gaps-1
            avnc = 0;
            avnd = 0;
            for day = 0:1:covid_days_gap-1
                avnc = avnc+table2array(covid_raw(current_day_index-covid_days_gap*group-day, 4));
                avnd = avnd+table2array(covid_raw(current_day_index-covid_days_gap*group-day, 5));
            end
            each_day = [each_day avnc/covid_days_gap avnd/covid_days_gap];
        end
    
        covid_suffix = [covid_suffix; each_day];
    end
    save("covid_suffix.mat", "covid_suffix");
    fprintf("Finished constructing covid_suffix.\n");
else
    fprintf("Found covid suffix.\n");
    load("covid_suffix.mat");
end

%% Construct the entirety of sample & label matrix
if ~isfile("sample_label_matrix.mat") || discard_trained_structures
    sample = [];
    fprintf("Constructing sample/label matrix...\n");
    tic
    
    total_stocks = max(stock_raw.Stock_ticker);
    for current_stock_id = 1:1:total_stocks
        % for each stock, based on their date range, custumize the sampling
        % date ranges
        current_stock = stock_raw(stock_raw.Stock_ticker==current_stock_id, :);
    
        datapoint_stock = [];
        fprintf("Stock id %d of %d...\n", current_stock_id, total_stocks);
        for current_day_index = 1+max(group_in_days_of*total_groups, covid_days_gap*covid_total_gaps)...
                :1:size(current_stock.Days, 1)-(label_days*label_groups)-1
            % for every stock day, produce a datapoint as sample
            current_day = current_stock.Days(current_day_index);
            if (~any(covid_suffix(:, 1)==current_day))
                continue
            end
            cd_stock = current_stock(current_day_index, :);
    
            datapoint = [cd_stock.Stock_ticker cd_stock.Section current_day];
            % 1 week of stock data in the past
            for into_past = 0:1:6
                pd_stock = current_stock(current_day_index-into_past, :);
                datapoint = [datapoint table2array(pd_stock(1, 2:6))];
            end
            % past stock data in groups, only care about average adjClose
            for group = 0:1:total_groups-1
                ave_adjC = 0;
                for day = 0:1:group_in_days_of-1
                    ave_adjC = ave_adjC+table2array(current_stock(current_day_index-group_in_days_of*group-day, 5));
                end
                datapoint = [datapoint ave_adjC/group_in_days_of];
            end
            % attach covid data
            datapoint = [datapoint covid_suffix(covid_suffix(:, 1)==current_day, :)];
            % construct the corresponding label
            future_ave_adjC = [];
            for future_group = 0:1:label_groups-1
                ave_adjC = 0;
                for into_future = 1:1:label_days
                    ave_adjC = ave_adjC+table2array(current_stock(current_day_index+into_future+future_group*label_days, 5));
                end
                future_ave_adjC = [future_ave_adjC ave_adjC/label_days];
            end
            datapoint = [datapoint future_ave_adjC];
    
            datapoint_stock = [datapoint_stock; datapoint];
        end
        sample = [sample; datapoint_stock];
    end
    
    label = sample(:, size(sample, 2)-label_groups+1:size(sample, 2));
    sample = sample(:, 1:size(sample, 2)-label_groups);
    save("sample_label_matrix.mat", "sample", "label")
    fprintf("Sample/label done!\n");
    toc
else
    fprintf("Found sample/label marix.\n")
    load("sample_label_matrix.mat")
end

%% seperate those from the future out -> train/valid/[test]
% In reality, the model will not have data avalible from the future to
% train; thus, data at the end (in terms of time) are
% removed from the train/test set to better simulate this.
% These will be later used to simulate "predicting future"
future_index = sample(:, 3)>(max(sample(:, 3))-(range(sample(:, 3))*back_protect));

sample_future = sample(future_index, :);
label_future = label(future_index, :);
sample(future_index, :) = [];
label(future_index, :) = [];

%% ML Model -> [train]/[valid]/test
% scrambled across stocks and dates
rng('default');
rng(1);
random_index = randperm(size(sample, 1));
sample_r = [];
label_r = [];
sample_r(random_index, :) = sample;
label_r(random_index, :) = label;

% seperate train/valid
last_train = floor(size(sample_r, 1)*train_per);
sample_train = sample_r(1:last_train, :);
sample_valid = sample_r(last_train+1:size(sample_r, 1), :);
label_train = label_r(1:last_train, :);
label_valid = label_r(last_train+1:size(label_r, 1), :);

forest = cell(label_groups, 1);
mape_progression = zeros(label_groups, 1);
if comp
    if isfile("trained_trees_comp.mat")
        fprintf("Found trained trees.\n");
        load("trained_trees_comp.mat");
    else
        fprintf("Training regression tree ensemble...\n");
    end
else
    if isfile("trained_trees.mat")
        fprintf("Found trained trees.\n");
        load("trained_trees.mat");
    else
        fprintf("Training regression tree ensemble...\n");
    end
end
tic
% Notice that this is training each prediction range seperatelly
% E.g. 1 week into the future is trained on a seperate tree than 
%      2 weeks into the future, eventhough the end result requires both
% This is for testing if *comp* improves later predictions by using the 
% predictions produced by earlier trees. 
for tree_index = 1:1:label_groups
    fprintf("Group %d of %d | ", tree_index, label_groups);
    if (comp)
        sample_train = [sample_train label_train(:, tree_index)];
        sample_valid = [sample_valid label_valid(:, tree_index)];
    else
        if (tree_index == 1)
            sample_train = [sample_train label_train(:, tree_index)];
            sample_valid = [sample_valid label_valid(:, tree_index)];
        else
            sample_train(:, size(sample_train, 2)) = label_train(:, tree_index);
            sample_valid(:, size(sample_valid, 2)) = label_valid(:, tree_index);
        end
    end
    if ~isfile("trained_trees.mat") || discard_trained_structures
        % best meta:
        % comp: method:LSBoost;cyc:488;rate:0.26789;min leaf size:14
        % no comp:     LSBoost;    433;     0.3595 ;               7
        tempTree = templateTree("MinParentSize", 1, "MinLeafSize", 14);
        % rtree = fitrensemble(sample_train(:, 1:size(sample_train, 2)-1), sample_train(:, size(sample_train, 2)),"Learners",tempTree,'OptimizeHyperparameters','auto');
        % pause;
        rtree = fitrensemble(sample_train(:, 1:size(sample_train, 2)-1), sample_train(:, size(sample_train, 2)), "NumLearningCycles", 488, ...
            "method", "LSBoost", "LearnRate", 0.26789, "Learners", tempTree);
    else
        rtree = cell2struct(forest(tree_index), 't').t;
    end
    prediction = predict(rtree, sample_valid(:, 1:size(sample_valid, 2)-1));
    mape_progression(tree_index) = mape(prediction, sample_valid(:, size(sample_valid, 2)));
    fprintf("MAPE = %.2f%%\n", mape_progression(tree_index));
    sample_valid(:, size(sample_valid, 2)) = prediction;
    % store the tree
    treeStruct = struct();
    treeStruct.t = rtree;
    forest(tree_index) = struct2cell(treeStruct);
end
if comp
    if ~isfile("trained_trees_comp.mat")
        save("trained_trees_comp.mat", "forest");
    end
else
    if ~isfile("trained_trees.mat")
        save("trained_trees.mat", "forest");
    end
end
sample_train(:, size(sample_future, 2)) = [];
sample_valid(:, size(sample_future, 2)) = [];
toc
figure;
plot(1:1:label_groups, mape_progression);
title("Validation MAPE");
xlabel(sprintf("Prediction groups (in business days of %d)", group_in_days_of));
ylabel('Mean Absolute Percentage Error');

%% Test on future stock prices
mape_progression_future = zeros(label_groups, 1);
mape_progression_future_mean = zeros(label_groups, 1); % the mean model
pred_coll = [];
fprintf("Testing with data from future...\n");
tic
for tree_index = 1:1:label_groups
    fprintf("Group %d of %d | ", tree_index, label_groups);
    if (comp)
        sample_future = [sample_future label_future(:, tree_index)];
    else
        if (tree_index == 1)
            sample_future = [sample_future label_future(:, tree_index)];
        else
            sample_future(:, size(sample_future, 2)) = label_future(:, tree_index);
        end
    end
    treeStruct = cell2struct(forest(tree_index), 't');
    treeStruct = treeStruct.t;
    prediction = predict(treeStruct, sample_future(:, 1:size(sample_future, 2)-1));
    pred_coll = [pred_coll prediction];
    mape_progression_future(tree_index) = mape(prediction, sample_future(:, size(sample_future, 2)));
    mape_progression_future_mean(tree_index) = mape(sample_future(:, 7), sample_future(:, size(sample_future, 2)));
    fprintf("MAPE = %.2f%% | ", mape_progression_future(tree_index));
    fprintf("RMSE = %.2f \n", rmse(prediction, sample_future(:, size(sample_future, 2))))
    if (comp)
        sample_future(:, size(sample_future, 2)) = prediction;
    end
end
sample_future(:, size(sample_future, 2)) = [];
toc
figure;
hold on;
plot(1:1:label_groups, mape_progression_future, 'DisplayName', "Prediction");
plot(1:1:label_groups, mape_progression_future_mean, 'DisplayName', "Mean Model");
name_comp = ["", "with comp"];
title(sprintf("Testing MAPE %s", name_comp(1+comp)));
legend({"Prediction", "Mean Model"}, 'Location', 'best');
xlabel(sprintf("Prediction groups (in business days of %d)", group_in_days_of));
ylabel('Mean Absolute Percentage Error');

%% Visual example of the prediction
pred_stock_id = 1;      % which stock(using id) to look for

sample_to_look_at = sample_future(sample_future(:, 1)==pred_stock_id, :);
last_day = max(sample_to_look_at(:, 3)); % day to be predicted for future
pred_to_look_at = pred_coll(sample_future(:, 1)==pred_stock_id, :);
pred_to_look_at = pred_to_look_at(sample_to_look_at(:, 3)==last_day, :);
stock_to_look_at = table2array(stock_raw(stock_raw.Stock_ticker==pred_stock_id, :));

starting_index = find(stock_to_look_at(:, 1)==last_day);
indices = starting_index+ceil(label_days/2)+(0:1:label_groups-1)*label_days;

figure;
hold on;
plot(stock_to_look_at(:, 1), stock_to_look_at(:, 5));
plot(stock_to_look_at(indices, 1), pred_to_look_at);
title(sprintf("Example of prediction on stock ID: %d %s", pred_stock_id, name_comp(1+comp)));
legend({"Real Stock Price", "Predicted Price"}, 'Location', 'best');
xlabel("Date");
ylabel("Price");
