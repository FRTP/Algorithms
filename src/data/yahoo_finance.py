# Usage:
# import yahoo_finance as yf
# yf.get_info(your_working_dir)

import datetime
import os
import pandas as pd
import urllib.request


class CChecker:
    def __init__(self, working_dir, shares_lst):
        self.working_dir = working_dir
        self.standart_filename = shares_lst[0] + ".csv"
        self.standart_dates = []
        fd = pd.read_csv(self.working_dir + self.standart_filename)
        self.standart_dates = fd.Date
        self.dates_for_remove = []

    def check(self, filename):
        fd = pd.read_csv(filename)
        volume = fd.Volume
        date = fd.Date
        for i in range(len(volume)):
            if int(volume[i]) == 0:
                self.dates_for_remove.append(date[i])
        new_set = set(date)
        standart_set = set(self.standart_dates)
        return new_set ^ standart_set

    def remove_zero_volumes(self):
        if len(self.dates_for_remove) == 0:
            return
        print("These dates are going to be deleted:\n"
              + str(self.dates_for_remove))
        file_list = filter(lambda x: x.endswith(".csv"),
                           os.listdir(self.working_dir))
        for zero_date in self.dates_for_remove:
            for filename in file_list:
                fd = pd.read_csv(self.working_dir + filename)
                fd = fd.drop(fd[fd['Date'] == zero_date].index)
                fd.to_csv(self.working_dir + filename, index=False)


start_date = 'a=00&b=01&c=2000'
now = datetime.datetime.now()

shares_list = [
        'INTC',
        'MDT',
        'MSFT',
        'AAPL',
        'ADBE',
        'CSCO',
        'EBAY',
        'EA',
        'NTAP',
        'NVDA',
        'ORCL',
        'SYMC',
        'YHOO',
        'XRX'
        ]


def open_url(share):
    end_date = "d=" + str(now.month - 1) + "&e=" + str(now.day) + \
            "&f=" + str(now.year)
    response = urllib.request.urlopen(
            'http://ichart.finance.yahoo.com/table.csv?s=' + share +
            '&' + start_date + '&' + end_date + '&g=d&ignore=.csv')
    return response.read()


def check_identity():
    standart_filename = shares_list[0] + ".csv"
    print("Standart file: " + standart_filename)


def get_info(working_dir):
    if working_dir[-1] != "/":
        working_dir += str("/")

    outfile = open(working_dir + shares_list[0] + ".csv", 'wb+')
    outfile.write(open_url(shares_list[0]))
    checker = CChecker(working_dir, shares_list)

    for i in range(1, len(shares_list)):
        filename = working_dir + shares_list[i] + ".csv"
        outfile = open(filename, 'wb+')
        outfile.write(open_url(shares_list[i]))
        test_set = checker.check(filename)
        if len(test_set) != 0:
            print("Difference in " + shares_list[i] + ".csv:\n"
                  + str(test_set))
            return
    checker.remove_zero_volumes()
