"""
db_interface.py:

to get data from database, with incremental period-retry
"""

import sys
import utils
import config
import psycopg2
import psycopg2.extras
from time import sleep
import pandas.io.sql as psql
from sqlalchemy import create_engine

def get_db_connection(conn_str):
    """
    Connects to database (5 retries with incremental delay)
    :param conn_str: the connection string as taken from config file
    :return: Database connection object
    """

    log = config.log

    retry = 1
    conn = None
    while conn is None and retry <= 5:
        try:
            conn = psycopg2.connect(conn_str)
        except psycopg2.Error as e:
            log.info("Database connection error: {0}\nCONNECTION STRING: {1}\nRetry {2}"
                .format(e, conn_str, retry))
            log.info("Wait {0} seconds until next retry..".format(retry * 10))
            from time import sleep
            sleep(retry * 10)  # Sleep for an incremental period (10, 20, 30 seconds)
            retry += 1

    if conn is None:
        log.error("Could not connect to the i-EM database")

    return conn



def do_simple_example_query() -> dict:
    """
    get the list of plants on public.controller_tabella_impianti
    :return: list of plants with their attributes
    """
    log = config.log
    rows = {}
    conn = get_db_connection(config.dba_connection_string)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = "SELECT * FROM public.controller_tabella_impianti"

    try:
        cur.execute(query)
        rows = cur.fetchall()
    except Exception as e:
        log.info("Error occurred while reading plants: {0} QUERY: {1}".format(e, query))
    finally:
        cur.close()
        conn.close()

    return rows



def do_pd_read_sql_example_query() -> dict:
    """
    get the list of plants on public.controller_tabella_impianti
    :return: list of plants with their attributes
    """
    log = config.log
    conn = get_db_connection(config.dba_connection_string)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = "SELECT nome_impianto, nome_descrittivo_impianto, latitude, longitude " \
            " FROM public.controller_tabella_impianti"

    try:
        log.debug("Connecting and querying...")
        df = psql.read_sql(query, conn, index_col=['nome_impianto'])
        log.debug("Dataframe created")

    except Exception as e:
        log.info("Error occurred while reading plants: {0} QUERY: {1}".format(e, query))
    finally:
        cur.close()
        conn.close()

    return df





def pd_to_sql_query(df, db_table_name,
                        engine_conn,
                        schema) -> int:
    """
    get the pandas dataframe and update it to database
    :param df: pandas dataframe to be insert. Assumed to be with same structure of the table in the postgreSQL database
    :param db_table_name: table name in the postgreSQL database                                             [string]
    :param engine_conn: engine connection                                                                   [string]
    :param schema: database schema                                                                          [string]
    :return: 0
    # ----------------------------------------------------------------------------------------------------------------
    .....................................................................................
    reference in http://docs.sqlalchemy.org/en/latest/core/engines.html 
    .....................................................................................
    # EXAMPLES:
    # to create an engine
    from sqlalchemy import create_engine
    engine_conn= 'postgresql://user:password@host:port/name'
    engine = create_engine(engine_conn)  
    OR, parameterizing:
    engine = create_engine('postgresql://{0}:{1}@{2}:{3}/{4}'
                               .format(user, password, host, port, name))
    # create engine:
    # dialect+driver://username:password@host:port/database
    # example:
    # default
    # engine = create_engine('postgresql://scott:tiger@localhost/mydatabase')
    # psycopg2
    # engine = create_engine('postgresql+psycopg2://scott:tiger@localhost/mydatabase')
    # ----------------------------------------------------------------------------------------------------------------

    """

    try:
        # creating log structure from config
        log = config.log

        # <editor-fold desc="region Database ">
        log.debug("Creating engine...")
        retry = 1
        engine = None
        while engine is None and retry <= 5:
            try:
                engine = create_engine(engine_conn)
                log.debug("Connecting and executing...")
                df.to_sql(db_table_name, engine, schema=schema, if_exists='append', chunksize=12*24*7)
                log.debug("Dataframe inserted")

            except Exception as e:
                log.info("Error occurred: {0}".format(e))
                log("Wait {0} seconds until next retry..".format(retry * 10))
                sleep(retry * 10)  # Sleep for an incremental period (10, 20, 30 seconds)
                retry += 1

        if engine is None:
            utils.send_error_msg_logfile_attach("Engine is none: error occurred")
            sys.exit(-1)  # This is a critical error: program will terminate

    except Exception as e:
        log.info("Error occurred: {0}".format(e))


    finally:
        del engine


    return 0


#example for testing pd_to_sql_spe03
#create a dataframe with an index and a values column
if 0:
    import pandas
    import numpy
    # create a dataframe with an index and a values column
    x = pandas.date_range('2013-01-01', '2013-01-07', freq='D')
    y = numpy.arange(len(x), dtype="float")
    y[int(len(x) / 3)] = numpy.NaN
    df = pandas.DataFrame(index=x, data=y, columns=['irradiance_raw'])
    df.index.name = "datetime"
    df['id_pmr'] = 919195
    df['tilt'] = 919193
    df['azimuth'] = 919191
    df['irradiance_pp'] = y
    df['sensor_name'] = "empty"
    for i in range(0, len(df)):
        df['sensor_name'][i] = str(i) + " Rolling ZZZZ"

    db_table_name = 'sasha_spe03_output'
    engine_conn = 'postgresql://postgres:iem01@localhost:5432/dba'
    schema = "meteorep"
    pd_to_sql_query(df, db_table_name, engine_conn, schema)

if 0:

    user = "postgres"
    password = 'iem01'
    host = "localhost"
    port = 5432
    name = "dba"
    schema = "public"

    engine_conn = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(user, password, host, port, name)

    import pandas
    spe03_test_infile = "//iemstaffstor.file.core.windows.net/res-dev/Sasha_data/sasha_output_apr2017/reports/" + "56924_sensor_check_report_SPE03_t24_a180.csv"
    df_spe03 = pandas.read_csv('{0}'.format(spe03_test_infile), index_col=0, sep=';')
    col_list = df_spe03.columns
    #example, real config to be taken from main code sensor_check_processing.py
    tilt_config = 30.0
    azim_config = 180.0
    id_pmr = 56924
    df_spe03.index.name='datetime'

    # <editor-fold desc="for region ">
    for i in (range(0,len(col_list),2)):
        sensor_name = col_list[i]
        print(sensor_name)
        df_temp_to_sql = df_spe03.iloc[:, i:i+2].copy()
        df_temp_to_sql.rename(columns={sensor_name: "irradiance_raw",
                                       (sensor_name + "_IRRADIANCE_pp"): "irradiance_pp"},
                              inplace=True)
        # adding a column with sensor name
        df_temp_to_sql["id_pmr"] = id_pmr
        df_temp_to_sql["tilt"] = tilt_config
        df_temp_to_sql["azimuth"] = azim_config
        df_temp_to_sql["sensor_name"] = sensor_name

        df_temp_to_sql.to_csv('//iemstaffstor.file.core.windows.net/res-dev/temp/prova_{0}.csv'.format(sensor_name),
                              sep=';', index=False)
        db_table_name = 'sasha_spe03_output'
        engine_conn = 'postgresql://postgres:iem01@localhost:5432/dba'
        schema = "meteorep"

        pd_to_sql_query(df_temp_to_sql, db_table_name, engine_conn, schema)
        del df_temp_to_sql
    # </editor-fold desc="for region ">









