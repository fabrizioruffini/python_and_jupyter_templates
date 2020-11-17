# <editor-fold desc="region general importing">
import os
import sys
import logging
from datetime import datetime
# </editor-fold desc="region general importing">


#  <editor-fold desc="region Mail configuration parameters">
smtp_server = 'some_server'
smtp_username = 'some_username'
smtp_password = 'some_pwd'
mail_sender = 'some_sender'
delivery_address = ['someone@somewhere.com']
reports_address = someone@somewhere.com']
debug_address = someone@somewhere.com']
#</editor-fold  "region Mail configuration parameters ">


#<editor-fold desc="region Database configuration parameters">
# Database configuration parameters
dba_host = some_host'
dba_name = 'some_dba'
dba_port = 5432
dba_user = 'some_user'
dba_password = 'some_pwd'

# Connection string to database containing plants info
dba_connection_string = "host = '{0}' dbname = '{1}' port = '{2}' user = '{3}' password = '{4}' application_name='Pacman'"\
    .format(dba_host, dba_name, dba_port, dba_user, dba_password)


# </editor-fold desc="region Database configuration parameters">


#<editor-fold desc="region logging setup">

# defining dir
log_dir = "{0}/{1}/".format(os.path.realpath(sys.path[0]) + '/log/', datetime.now().strftime("%Y-%m"))
if not os.path.exists(log_dir): os.makedirs(log_dir)

# defining log name
log_name = os.path.basename(sys.argv[0])

# defining log configuration
logfile_complete_name = '{0}{2}_{1}.log'.format(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"), log_name)
logging.basicConfig(filename=logfile_complete_name,
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S%z]')

log = logging.getLogger(log_name)

# define a Handler which writes messages to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='[%H:%M:%S%z]')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


#</editor-fold desc="region logging setup">


#<editor-fold desc="region setting input and output dirs">

#indir = './input/'
#if not os.path.exists(indir): os.makedirs(indir)

#outdir = './output/'
#if not os.path.exists(outdir): os.makedirs(outdir)

figdir = './fig/'
if not os.path.exists(figdir): os.makedirs(figdir)

#</editor-fold desc="region setting input and output dirs">



#<editor-fold desc="region setting pooling">
pool_size = 2
#</editor-fold desc="region setting pooling">
