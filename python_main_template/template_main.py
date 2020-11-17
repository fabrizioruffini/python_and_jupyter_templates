"""
usage --> python __main__.py :


SW name:

This software evaluates ...

Library dependencies:
ex: psycopg2, pytz, iem_stat (requires numpy), iem_utilities
"""


# <editor-fold desc="region general importing">
import os
import sys
import time
import pandas
import socket
import argparse
import pkg_resources
from multiprocessing import Pool
from datetime import datetime, date, timedelta
# </editor-fold desc="region general importing">

# <editor-fold desc="region  specific i-EM importing">
import utils
import config
import db_interface
import example_func
from iem_plotlib.iem_style import Init_iem_style
from iem_plotlib.iem_style import clear_iem_style
# </editor-fold desc="region general importing">


__author__ = 'fabrizio.ruffini'
__version__ = '0.0.1'
__date__ = 'Date of creation 13/03/2017'
__status__ = 'prealpha'


def main(args_main):
    """
    Main function:
    This is a template main function. 
    :param args_main
    :return: 0 if OK, -1 otherwise
    """

    try:

        # <editor-fold desc="region Import logging and print basic info">

        log = config.log
        # print The program-is-starting info
        log.info('\n#########      Program Started      ##########')

        log.info("\n-----------BEGIN LOGGIN BASIC PROGRAM INFO-----------")
        log.info('Setting iem_plot_style')
        Init_iem_style()

        # print the host info (the computer running the script)
        log.info("hostname: {0}".format(socket.gethostname()))
        #print the Platform identifier
        log.info("Platform identifier: {0}".format(sys.platform))
        # print the software version
        log.info('Software Version: {0}'.format(__version__))
        # print The timeoffset wrt to UTC: the computer could have a local timezone  different from UTC
        log.info("UTC time: {0}, Difference with local time = {1}".format(datetime.utcnow().strftime("%H:%M"), (datetime.now() - datetime.utcnow())))
        # print external arguments of the software
        log.info("Argparse Configuration: ")
        log.info(args_main)
        log.info('Python version: {0}'.format(sys.version))
        log.info("Working directory: {0}".format(os.getcwd()))
        #pythonpath
        full_pythonpath =[]
        for p in sys.path:
            full_pythonpath.append(p.strip())
        log.info("pythonpath: {}".format(full_pythonpath))
        log.info("\n-----------END LOGGIN BASIC PROGRAM INFO-----------")

        # </editor-fold desc="region Import logging and print basic info">

        # <editor-fold desc="region Example query read from db_interface">
        # df = db_interface.do_pd_read_sql_example_query()
        # log.debug(df.head(3))
        # </editor-fold desc="region Example query read from db_interface">

        # <editor-fold desc="region Example query write to db_interface">
        # get a df with same structure of postgresql table
        # log.debug(df.head(3))  #take a look at data
        # copy to db
        # df = db_interface.pd_to_sql_example_query()
        # </editor-fold desc="region Example query write to db_interface">


        # <editor-fold desc="region Example multiprocessing">
        # if args.mode == 'debug':
        #    pool_size = 1
        # else:
        #   pool_size = config.pool_size

        # process_names = ['a', 'b', 'c']
        # arguments_list = []
        # for name in process_names:
            # an appending is needed to create a dictionary for each processing
            # arguments_list.append({'arg_1': name, 'arg_2': 'string_2'})

        # pool = Pool(processes=pool_size)
        # result = pool.map_async(example_func.func_to_be_parallelized, arguments_list)
        # Wait until every task has finished
        # while True:
        #     if result.ready():
        #         break
        #     time.sleep(0.1)
        #
        # results_list = result.get()
        #
        # log.info(results_list)
        #
        # </editor-fold desc="region Example multiprocessing">
        #
        # <editor-fold desc="region TEST Error stacking for emailing">
        # Test_raising_an_exception_Eccezziunale_veramente
        # </editor-fold desc="region TEST Error stacking for emailing">
        #

        log.info('\nResetting plot_style to matplotlib default')
        clear_iem_style()

    # end try



    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname=os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        utils.error("line {0}; FR: some unexpected error occurred in script {1}: {2}".format(exc_tb.tb_lineno, fname, err))
        log.info('######### Program Ended with Errors; sending mail #########')

        return None


    log.info('\n######### Program Ended successfully #########')

    return 0




if __name__ == '__main__':
    # Change current working directory to software directory

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Argument parsing
    parser = argparse.ArgumentParser(description='description:\n  Main\n'
                                                 'This sw does this and that ...',
                                     usage='\n  __main__.py \n',
                                     formatter_class=argparse.RawTextHelpFormatter)

    date_default = str(date.today() - timedelta(days=15))
    parser.add_argument('-sd', '--start_date',
                        default=date_default,
                        help='Start date (included). Default: {0}'.format(str(date_default)))

    parser.add_argument('-mode', '--mode',
                        default='debug',
                        help='modality')

    parser.add_argument('-stage', '--stage',
                        default='development',
                        help='Stage: can be in development, staging, production')


    args = parser.parse_args()

    main(args)
    #utils.send_error_msg(args.mode)
    utils.send_error_msg_logfile_attach(os.path.basename(sys.argv[0]))
