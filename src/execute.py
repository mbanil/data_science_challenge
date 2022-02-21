#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession

if __name__ == "__main__":
    try:
        start_time = datetime.now() 

        LOG.info('Begin execution')

        args = get_args()

        LOG.info('Arguments:\n{}'.format(args))
        
        spark = SparkSession.builder.appName(args.app_name).getOrCreate()


        data = data_extraction(args.training_data_filename, args)
        data = data_preprocessing(data)
        store_schema(data, args.schema_filename)
        data = feature_extraction(data, args)
        model = train(spark, data, args)
        spark.stop()        
        
        writeLogFile(args.logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: Execution finished !!")   
        
        print("total time taken in seconds ",  (datetime.now() - start_time).total_seconds())
        
        writeLogFile(args.logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Total time taken in seconds : " + str((datetime.now() - start_time).total_seconds()))   

        ####################### Garbage collection ####################################
        collected = gc.collect()
        
    except Exception as e:
        writeLogFile(args.error_logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: "+  str(e))        
































import os
import gc
import warnings
warnings.filterwarnings("ignore")
from args import get_args
from datetime import datetime
from Logs.logger import writeLogFile

from Utilities.training import train
#from Utilities.data_validation import get_data_validation_report
from Utilities.utility import data_extraction, data_preprocessing, feature_extraction, store_schema


if __name__ == "__main__":
    try:
        start_time = datetime.now() 
        args = get_args()
        writeLogFile(args.logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: Execution started !!")   
        
        spark = SparkSession.builder.appName(args.app_name).getOrCreate()
        data = data_extraction(args.training_data_filename, args)
        data = data_preprocessing(data)
        store_schema(data, args.schema_filename)
        data = feature_extraction(data, args)
        model = train(spark, data, args)
        spark.stop()        
        
        writeLogFile(args.logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: Execution finished !!")   
        
        print("total time taken in seconds ",  (datetime.now() - start_time).total_seconds())
        
        writeLogFile(args.logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Total time taken in seconds : " + str((datetime.now() - start_time).total_seconds()))   

        ####################### Garbage collection ####################################
        collected = gc.collect()
        
    except Exception as e:
        writeLogFile(args.error_logfile, datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: "+  str(e))        