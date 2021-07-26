#!/bin/bash

excel_file=$1
prefix=$2

function plot() {
    python -m analysis.power $excel_file $1 $2 $prefix$1.pdf
}

# plot mobilenet_uci-har profiled-confs/conf-mobilenet_uci-har-5000,3.0,10.0-200-cost_linear,.all.android-profiled.50+400+monsoon+take2.txt
# plot mobilenet_cifar10 profiled-confs/conf-mobilenet_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.android-profiled.100+800+monsoon.txt
# plot alexnet2_cifar10 profiled-confs/conf-alexnet2_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.android-profiled.100+800+monsoon.txt
# plot vgg16_cifar10 profiled-confs/conf-vgg16_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.android-profiled.25+200+monsoon.txt
# plot resnet50_uci-har profiled-confs/conf-resnet50_uci-har-5000,3.0,5.0-50-cost_linear,.edited.android-profiled.50+250+monsoon+take2.txt

# plot mobilenet_uci-har /home/matevz/Documents/MAG/hpvm-tuning-android/profiled-confs/conf-mobilenet_uci-har-5000,3.0,10.0-200-cost_linear,.filtered.android-profiled.for-confidence-full.txt

plot mobilenet_uci-har ../tuned-configurations/conf-mobilenet_uci-har-5000,3.0,10.0-200-cost_linear,.all.txt
plot mobilenet_cifar10 ../tuned-configurations/conf-mobilenet_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.txt
plot alexnet2_cifar10 ../tuned-configurations/conf-alexnet2_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.txt
plot vgg16_cifar10 ../tuned-configurations/conf-vgg16_cifar10-10000,3.0,5.0-50-cost_linear,qos_p1.all.txt
plot resnet50_uci-har ../tuned-configurations/conf-resnet50_uci-har-5000,3.0,5.0-50-cost_linear,.all.txt
