#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logger import LOG
from args import get_args
from utils.utils import data_preprocessing, data_extraction, store_schema
from utils.training import train

from datetime import datetime

from pyspark.sql import SparkSession

from utils.utils import remove_outliers

if __name__ == "__main__":
    # try:
    start_time = datetime.now() 

    LOG.info('Begin execution')

    args = get_args()

    LOG.info('Arguments:\n{}'.format(args))

    LOG.info('Starting spark session')
    
    spark = SparkSession.builder.appName(args.app_name).getOrCreate()

    data = data_extraction(spark, args.data_filename, args)
    data = data_preprocessing(data, args)

    # store_schema(data, args.schema_filename)

    train(spark, data, args)

    spark.stop()
