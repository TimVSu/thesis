# thesis
bachelor-thesis repository

## How to get it running:
- install required packages (pip install -r requirements.txt)
- For data collection you need a need to create a reddit API app and save the resulting credentails "CLIENT_ID", "CLIENT_SECRET" and "USER_AGENT" in your environement variables

## using the script:
The program has three main functions, creating a textual dataset, creating a timeseries dataset and running a BERT classifier
- to create a textual dataset run python main.py --create_text_DS and specify --test_dataset_dir "path to directory for the test dataset" and --train_dataset_dir "path to directory for the train dataset"
- to create a time series dataset run --create_TS_DS and specify --timeseries_filename "basename of timeseries train and test dataset files", --timeseries_directory "directory to save the timeseries datasets in", --timeseries_type "finegraining and timespan of time series dataset", options are "dayHour",  "dayMinute", "weekHour", "weekMinute"
  - The result are a timeseries_filename_TRAIN.tsv and timeseries_filename_TEST files that fit the structure required by the framework of Fawaz et al. (https://github.com/hfawaz/dl-4-tsc.git)
- To run a BERT classifier run python main.py --run_BERT and specify --test_dataset_dir "directory of the test dataset" and --train_dataset_dir "directory of the training dataset"
- Further customization such as a grid search, specifying hyperparameters etc. are available, run python main.py --help for more info