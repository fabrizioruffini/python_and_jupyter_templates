__author__ = '[fabrizio.ruffini, giulia.pinnisi]'

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email.encoders import encode_base64
from datetime import datetime
import os

def sendemail(from_addr, to_addr_list="None", cc_addr_list=[],
              subject="None", message="None",
              login="None", password="None",
              smtpserver="None"):
    '''
    ------------------------------------------------------
    mail sending function. Takes as input:
    from_addr:    sender address
    to_addr_list: receivers address list
    to_addr_list: cc receivers address list
    subject
    message
    login
    password
    ------------------------------------------------------
    '''
    header  = 'From: ' + from_addr + '\n'
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'CC: %s\n' % ','.join(cc_addr_list)
    header += 'Date: {0}\n'.format(formatdate(localtime=True))
    header += 'Subject: ' + subject + ' \n\n'
    message = header + message

    try:
        print("Sending email to %s" % ','.join(to_addr_list))
        server = smtplib.SMTP(smtpserver)
        server.ehlo()
        server.starttls()
        server.login(login, password)
        full_add_list = to_addr_list + cc_addr_list
        mail = server.sendmail(from_addr, full_add_list, message)
        server.quit()
        print("Mail successfully sent!")
        return mail

    except Exception as e:
        print("Error: unable to send email: {0}".format(e))

		
def sendemail_with_attachment(smtp_server, smtp_username, smtp_password, sender_address, receiver_address_list,
                              ccn_address_list, file_path, subject, message_body) -> bool:
    """
    Sends an email with a file attached
    :param smtp_server: SMTP server
    :param smtp_username: SMTP username
    :param smtp_password: SMTP password
    :param sender_address: Sender address (string)
    :param receiver_address_list: Receiver address list (list of strings)
    :param ccn_address_list: CCN address list (list of strings)
    :param file_path: Absolute file path to the file to attach
    :param subject: Mail subject
    :param message_body: Mail body
    :return: True if mail sended, False otherwise
    """
    # Check input type
    if type(smtp_server) is not str or \
       type(smtp_username) is not str or \
       type(smtp_password) is not str or \
       type(sender_address) is not str or \
       type(receiver_address_list) is not list or \
       type(ccn_address_list) is not list or \
       type(file_path) is not str or \
       type(subject) is not str or \
       type(message_body) is not str:
        raise TypeError("Arguments type does not match prototype")

    log("Sending email to {0}".format(receiver_address_list))
    log("CCN: {0}".format(ccn_address_list))
    msg = MIMEMultipart()
    msg['From'] = sender_address
    msg['To'] = COMMASPACE.join(receiver_address_list)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(message_body))
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(file_path, "rb").read())
    encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file_path))
    msg.attach(part)

    # Send message
    try:
        smtp = smtplib.SMTP(smtp_server)
        smtp.login(smtp_username, smtp_password)
        # Adding delivery address here but not in 'To' field causes it to be in 'BCC'
        smtp.sendmail(sender_address,  receiver_address_list + ccn_address_list, msg.as_string())
        smtp.close()
    except Exception as e:
        log("Error occurred while sending email error: {0}".format(e))
        return False

    log("Mail sent")
    return True


def log(message):
    """
    Print message on terminal adding timestamp
    :param message: Message to print
    """
    print(('[{0}]: ' + message).format(datetime.utcnow().strftime("%H:%M:%S")))
