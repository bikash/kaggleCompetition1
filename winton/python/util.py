import time

__author__ = 'Gavin.Chan'


def set_timer(start=time.gmtime(0)):
    if start == time.gmtime(0):
        start = time.localtime()
        print("Start time = %s" % time.strftime("%c", start))
    else:
        end = time.localtime()
        print("Run time = %s" % (time.mktime(end) - time.mktime(start)))
    return start

if __name__ == '__main__':
    start = set_timer()
    set_timer(start)
