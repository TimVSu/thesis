import fineTunedBert
import reddit_scraper
import utils
import numpy as np
import pandas as pd
import argparse
import timeSeriesDsCreator

TRAIN_DATASETS = './train_datasets/'
TEST_DATASEST = './test_datasets/'

MODEL_MAP = fineTunedBert.MAP_NAME_TO_HANDLE
PREPROCESS_MAP = fineTunedBert.MAP_MODEL_TO_PREPROCESS

parser = argparse.ArgumentParser(
    prog='thesis-reddit-classifier',
    description='Program for a bachelors-thesis about detecting ADHD on Reddit',
    epilog=''
)
parser.add_argument('--run_BERT', action='store_true')
parser.add_argument('--BERT-type', default='small_bert/bert_en_uncased_L-2_H-768_A-12')
parser.add_argument('--grid_search', action='store_true')
parser.add_argument('--gs_output_file')
parser.add_argument('-e', '--epochs', default=3, type=int)
parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch-size', default=32, type=int)
parser.add_argument('-cbs', '--comment-batch-size', default=100)
parser.add_argument('-cl', '--comment_limit', default=350, type=int)
parser.add_argument('--fixed_cbs', default=False)
parser.add_argument('--test_dataset_dir')
parser.add_argument('--train_dataset_dir')
parser.add_argument('--create_TS_DS', action='store_true')
parser.add_argument('--timeseries_filename')
parser.add_argument('--create_text_DS', action='store_true')
parser.add_argument('--timeseries_type', choices=['dayHour', 'dayMinute', 'weekHour', 'weekMinute'])
parser.add_argument('--DS_size', default=500, type=int)
parser.add_argument('--test_DS_size', default=100, type=int)
parser.add_argument('--keywords', default=['adhd', 'ADHD'])
parser.add_argument('--target_subreddit', default='ADHD')
parser.add_argument('--non_target_subreddits', default=['askreddit', 'worldnews', 'funny', 'aww', 'todayilearned'])
parser.add_argument('--comparison_subreddits', default=['ADHD', 'adhdwomen', 'adhdmeme'])
args = parser.parse_args()

if args.create_TS_DS:
    if all(hasattr(args, attr) for attr in ['timeseries_filename', 'timeseries_type']):
        timestampTuples = reddit_scraper.createTimeSeriesList(args.target_subreddit, args.non_target_subreddits, 
        args.DS_size, args.comparison_subreddits, args.keywords, 50, args.comment_batch_size, args.fixed_cbs)
        timeSeriesDsCreator.createTimeSeriesDataset(timestampTuples, args.timeseries_filename, args.timeseries_type)
    else:
        print('you have to specify a file name and timseries type for the time series dataset (--timeseries_filename and --timeseries_type)')

if args.create_text_DS:
    if all(hasattr(args, attr) for attr in ['test_dataset_dir', 'train_dataset_dir']):
        reddit_scraper.fillDirectoryTextual(args.target_subreddit, args.non_target_subreddits, args.DS_size, args.train_dataset_dir, args.test_dataset_dir, 
        args.comparison_subreddits, args.keywords, args.comment_limit, args.comment_batch_size, args.fixed_cbs)
        reddit_scraper.fillDirectoryTextual(args.target_subreddit, args.non_target_subreddits, args.test_DS_size, args.test_dataset_dir, args.train_dataset_dir, 
        args.comparison_subreddits, args.keywords, args.comment_limit,  args.comment_batch_size, args.fixed_cbs)
    else: print('you have to specify paths to the test and train dataset directorys (--test_dataset_dir and --train_dataset_dir')

if args.run_BERT:
    if args.grid_search:
        if all(hasattr(args, attr) for attr in ['test_dataset_dir', 'train_dataset_dir', 'gs_output_file']):
            utils.bertGridSearch(args.train_dataset_dir, args.test_dataset_dir, args.gs_output_file, args.BERT_type)
        else:
            print('you need to specify an output file for the grid search and the location of train and test datasets (--gs_output_file, --train_dataset_dir, _test_dataset_dir')
    else:
        if all(hasattr(args, attr) for attr in ['test_dataset_dir', 'train_dataset_dir']):
            fineTunedBert.bertRunner(args.train_dataset_dir, args.test_dataset_dir, args.batch_size, 32, args.BERT_type, MODEL_MAP, PREPROCESS_MAP, args.epochs, args.learning_rate)
        else:
            print('you need to specify train and test dataset locations to run BERT (--train_dataset_dir, --test_dataset_dir)')