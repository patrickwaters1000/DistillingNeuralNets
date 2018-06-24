#!/bin/bash
url=$1
save_path=$2
curl $url --max-time 10 > $save_path
