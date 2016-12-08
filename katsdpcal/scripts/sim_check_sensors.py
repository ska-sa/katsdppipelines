#! /usr/bin/env python

from katcp import BlockingClient, Message
import time

client = BlockingClient('', 5000)
client.start()

# print a list of all of the sensors
print 'Sensor list'
time.sleep(0.5)
sensor_list = client.blocking_request(Message.request("sensor-list"))[1]
for s in sensor_list:
    print '  ', s
print

# read the accumulator index sensor every two seconds
while True:
    time.sleep(2.)
    print client.blocking_request(Message.request("sensor-value", 'accumulator-indices'))
