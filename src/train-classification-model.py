
from logger import LOG
from args import get_args
from utils import utils
from utils.training import train

from datetime import datetime


from utils.utils import remove_outliers

if __name__ == "__main__":

    try:
        
        LOG.info('Begin execution')

        args = get_args()

        LOG.info('Arguments:\n{}'.format(args))

        data = utils.data_extraction(args.data_filename)
        data = utils.data_preprocessing(data, args)
        data = utils.encode_data(data, args.columns_to_encode)
        utils.store_schema(data, args.schema_path)
        train(data, args)

        LOG.info('End execution')


    except Exception as e:
        LOG.exception("Exception occured: " + e)

