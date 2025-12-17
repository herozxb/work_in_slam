#!/bin/bash


pid=`ps -ef | grep pv-lio* | grep -v grep | awk '{print $2}'`
kill -9 $pid

exit

