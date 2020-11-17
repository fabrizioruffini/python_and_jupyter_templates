import sys, os
import pip
import config
import platform
from datetime import datetime
from iem_utilities.mailing import sendemail, sendemail_with_attachment

#additional functions
# Error log
error_log = ""


def error(message):
    """
    Print error message and save it in error log.
    Error log will be sent at the end of execution.
    :param message: error message
    """
    log = config.log
    log.error("ERROR: " + message)

    global error_log
    error_log += "[{0}] ERROR OCCURRED:\n{1}\n\n\n".format(datetime.now().strftime("%H:%M:%S"), message)


def send_error_msg(mode: str) -> bool:
    """
    Sends an error message via email to software reports address.
    :param mode: Name_of_program mode
    :return: True: all ok, False: some errors occurred in mail sending
    """

    subject = '[{0}, mode: {0}] Error occurred while processing data'.format(os.path.basename(sys.argv[0]), mode)

    receivers_address_list = config.reports_address

    global error_log
    body = '{0}, Hostname: {1}, Platform: {2} \nPython version: {3}'.format(os.path.basename(sys.argv[0]), platform.node(), platform.platform(), platform.python_version())
    body += '\nPython packages: {0}'.format(sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]))
    body += "\n\n" + error_log
    print("error_log = ", error_log)
    if error_log != '':
        error_log = ''
        return sendemail(config.mail_sender, receivers_address_list, [], subject, body, config.smtp_username,
                         config.smtp_password, config.smtp_server)  #aggiungi log
    else:
        return True



def send_error_msg_logfile_attach(mode: str) -> bool:
    """
    Sends an error message + logifle attached via email to software reports address.
    :param mode: Name_of_program mode
    :return: True: all ok, False: some errors occurred in mail sending
    """

    subject = '[Name_of_program {0}] Error occurred while processing data'.format(mode)

    receivers_address_list = config.reports_address

    global error_log
    body = 'Name_of_program Hostname: {0}, Platform: {1} \nPython version: {2}'.format(platform.node(), platform.platform(), platform.python_version())
    #body += '\nPython packages: {0}'.format(sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]))
    body += "\n\n" + error_log
    if error_log != '':
        error_log = ''
        return sendemail_with_attachment(sender_address=config.mail_sender, smtp_username=config.smtp_username,
         smtp_server=config.smtp_server, smtp_password=config.smtp_password, receiver_address_list=receivers_address_list,
         ccn_address_list = config.debug_address,
         subject=subject, message_body=body,
        file_path= config.logfile_complete_name)
    else:
        return True



def send_end_msg(mode: str, text_msg: str) -> bool:
    """
    Sends a report message via email to software reports address.
    :param mode: Name_of_program mode
    :param text_msg: text to be sent
    """

    subject = '[Name_of_program {0}] Result'.format(mode)

    receivers_address_list = config.reports_address

    body = 'Name_of_program Hostname: {0}, Platform: {1} \nPython version: {2}'.format(platform.node(), platform.platform(), platform.python_version())
    body += '\nPython packages: {0}'.format(sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]))
    body += "\n\n" + text_msg
    return sendemail(config.mail_sender, receivers_address_list, [], subject, body, config.smtp_username,
                     config.smtp_password, config.smtp_server)



