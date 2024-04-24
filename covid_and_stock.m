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

%% Feature setup
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
% Back_protect: these are removed from the train/test set to simulate using
% the model on newly acquired data. 
group_in_days_of = 7;
total_groups = 8;
covid_days_gap = 7;
covid_total_gaps = 8;

norm_switch = false;

label_days = 7;
label_groups = 12;

train_per = 0.8;
back_protect = 0.25;

% model structure: later layers do(T)/DO NOT(F) use prediction from privious layers
comp = false;

%% Import: covid\us_epidemiology.csv
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Date", "Cases", "Deaths", "New_cases", "New_deaths", "Avg_new_cases", "Avg_new_deaths"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");
usepidemiology = readtable("covid\us_epidemiology.csv", opts);
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
usstock = readtable("stocks\us_stock.csv", opts);
clear opts

%% Convert dates to "days from 2000-01-01" (Y/M/D)
%  and tickers to id's. Everything have to be vectors.
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

%% construct and reuse the covid suffixes
covid_suffix = [];
day_start = min(covid_raw.Days)+8 ...
    +covid_days_gap*covid_total_gaps;
day_end = max(covid_raw.Days);

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
fprintf("Finished constructing covid_suffix.\n");

%% the entirety of the sample & label matrix
tic
sample = [];
total_stocks = max(stock_raw.Stock_ticker);
for current_stock_id = 1:1:total_stocks
    % for each stock, based on their date range, custumize the sampling
    % date ranges
    current_stock = stock_raw(stock_raw.Stock_ticker==current_stock_id, :);

    day_start = max(min(current_stock.Days), min(covid_raw.Days))+8+2*floor(group_in_days_of*total_groups/7) ...
        +max(group_in_days_of*total_groups, covid_days_gap*covid_total_gaps);
    day_end = min(max(current_stock.Days), max(covid_raw.Days))-floor((7+1)*label_days*label_groups/5);

    datapoint_stock = [];
    fprintf("Stock id %d of %d...\n", current_stock_id, total_stocks);
    for current_day_index = 1:1:size(current_stock.Days, 1)
        if (current_stock.Days(current_day_index)<day_start)
            continue;
        elseif (current_stock.Days(current_day_index)>day_end)
            break;
        end
        % for every stock day, produce a datapoint as sample
        current_day = current_stock.Days(current_day_index);
        cd_stock = current_stock(current_day_index, :);

        datapoint = [cd_stock.Stock_ticker cd_stock.Section current_day];
        % 1 week of stock data
        for into_past = 0:1:6
            pd_stock = current_stock(current_day_index-into_past, :);
            datapoint = [datapoint table2array(pd_stock(1, 2:6))];
        end
        % past stock data in groups
        for group = 0:1:total_groups-1
            ave_adjC = 0;
            for day = 0:1:group_in_days_of-1
                ave_adjC = ave_adjC+table2array(current_stock(current_day_index-group_in_days_of*group-day, 5));
            end
            datapoint = [datapoint ave_adjC/group_in_days_of];
        end
        % attach covid data
        datapoint = [datapoint covid_suffix(covid_suffix(:, 1)==current_day, :)];
        % sample label
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

future_index = sample(:, 3)>(max(sample(:, 3))-(range(sample(:, 3))*back_protect));

mu = mean(sample, 1);
sigma = std(sample, 0, 1);
if norm_switch 
    sample = (sample - mu) ./ sigma;
end

label = sample(:, size(sample, 2)-label_groups+1:size(sample, 2));
sample = sample(:, 1:size(sample, 2)-label_groups);
fprintf("Job's done!\n");
toc

%% seperate those from the future out -> train/test/[prove]
% In reality, the model will not have data avalible from the future to
% train; thus, data at the end (in terms of time) are
% removed from the train/test set to better simulate this.
% These will be later used to simulate "predicting future"
sample_future = [];
label_future = [];
sample_future = sample(future_index, :);
label_future = label(future_index, :);
sample(future_index, :) = [];
label(future_index, :) = [];

%% ML Model -> [train]/[test]/prove
% scramble
rng('default');
rng(1);
random_index = randperm(size(sample, 1));
sample_r = [];
label_r = [];
sample_r(random_index, :) = sample;
label_r(random_index, :) = label;

% seperate train/test
last_train = floor(size(sample_r, 1)*train_per);
sample_train = sample_r(1:last_train, :);
sample_test = sample_r(last_train+1:size(sample_r, 1), :);
label_train = label_r(1:last_train, :);
label_test = label_r(last_train+1:size(label_r, 1), :);

% best meta: method: LSBoost; cyc: 451; rate: 0.32997; min leaf size: 7
tempTree = templateTree("MinParentSize", 1, "MinLeafSize", 7);
forest = cell(label_groups, 1);
mape_progression = zeros(label_groups, 1);
fprintf("Training regression tree ensemble...\n");
tic
for tree_index = 1:1:label_groups
    fprintf("Group %d of %d | ", tree_index, label_groups);
    if (comp)
        sample_train = [sample_train label_train(:, 1)];
        label_train(:, 1) = [];
        sample_test = [sample_test label_test(:, 1)];
        label_test(:, 1) = [];
    else
        if (tree_index == 1)
            sample_train = [sample_train label_train(:, 1)];
            label_train(:, 1) = [];
            sample_test = [sample_test label_test(:, 1)];
            label_test(:, 1) = [];
        else
            sample_train(:, size(sample_train, 2)) = label_train(:, 1);
            label_train(:, 1) = [];
            sample_test(:, size(sample_test, 2)) = label_test(:, 1);
            label_test(:, 1) = [];
        end
    end
    rtree = fitrensemble(sample_train(:, 1:size(sample_train, 2)-1), sample_train(:, size(sample_train, 2)), "NumLearningCycles", 451, ...
        "method", "LSBoost", "LearnRate", 0.33, "Learners", tempTree);
    prediction = predict(rtree, sample_test(:, 1:size(sample_test, 2)-1));
    mape_progression(tree_index) = mape(prediction, sample_test(:, size(sample_test, 2)));
    fprintf("MAPE = %.2f%%\n", mape_progression(tree_index));
    sample_test(:, size(sample_test, 2)) = prediction;
    % store the tree
    treeStruct = struct();
    treeStruct.t = rtree;
    forest(tree_index) = struct2cell(treeStruct);
end
toc
figure;
plot(1:1:label_groups, mape_progression);

%% Future proof
mape_progression_future = zeros(label_groups, 1);
fprintf("Testing with data from future...\n");
tic
for tree_index = 1:1:label_groups
    fprintf("Group %d of %d | ", tree_index, label_groups);
    if (comp)
        sample_future = [sample_future label_future(:, 1)];
        label_future(:, 1) = [];
    else
        if (tree_index == 1)
            sample_future = [sample_future label_future(:, 1)];
            label_future(:, 1) = [];
        else
            sample_future(:, size(sample_future, 2)) = label_future(:, 1);
            label_future(:, 1) = [];
        end
    end
    treeStruct = cell2struct(forest(tree_index), 't');
    treeStruct = treeStruct.t;
    prediction = predict(treeStruct, sample_future(:, 1:size(sample_future, 2)-1));
    mape_progression_future(tree_index) = mape(prediction, sample_future(:, size(sample_future, 2)));
    fprintf("MAPE = %.2f%% | ", mape_progression_future(tree_index));
    fprintf("RMSE = %.2f \n", rmse(prediction, sample_future(:, size(sample_future, 2))))
    if (comp)
        sample_future(:, size(sample_future, 2)) = prediction;
    end
end
toc
figure;
plot(1:1:label_groups, mape_progression_future);

