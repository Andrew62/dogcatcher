# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:26:35 2016

@author: andrew
"""

import smtplib
from config import workspace
from email.mime.text import MIMEText


class MailMe(object):
    def __init__(self, user, password, **kwargs):
        self.__user__ = user
        self.__password__ = password
        self.__provider__ = kwargs.pop('provider', "smtp.mail.yahoo.com")
        self.__provider_port__ = kwargs.pop('provider_port', 587)
        
        
    def send_message(self, to, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.__user__
        msg['To'] = to
        try:
            server = smtplib.SMTP(self.__provider__, self.__provider_port__)
            server.starttls()
            server.login(self.__user__,self.__password__)
            server.sendmail(self.__user__, to, msg.as_string())
            server.quit()
        except Exception as e:
            raise e

def send_mail(subj, msg):        
    username = workspace.from_email 
    password = workspace.password  
    mailer = MailMe(username, password)
    mailer.send_message(workspace.to_email, subj, msg)

   
