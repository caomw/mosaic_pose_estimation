#!/usr/bin/python

"""
Copyright (c) 2012,
Systems, Robotics and Vision Group
University of the Balearican Islands
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Systems, Robotics and Vision Group, University of 
      the Balearican Islands nor the names of its contributors may be used to 
      endorse or promote products derived from this software without specific 
      prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

PKG = 'mosaic_pose_extractor' # this package name

import subprocess
import time
import roslib; roslib.load_manifest(PKG)
import rospy
import os
import sys
import argparse
import signal

def cut_bag(inbags, start_time, duration):
  cut_name = '/tmp/cut_bag.bag'
  cut_cmd = ['rosrun', 'bag_tools', 'cut.py', '--inbag']
  cut_cmd += inbags
  cut_cmd += ['--outbag', cut_name, '--start', str(start_time - duration / 2.0), '--duration', str(duration)]
  print '=== cutting input bagfile(s):', ' '.join(cut_cmd)
  subprocess.check_call(cut_cmd)
  return cut_name
 
def getGroundTruth(inbags, start_time, duration, output, camera, param_file):
  # cut_bag_name = cut_bag(inbags, start_time, duration)
  left_image_topic  = camera + '/left/image_raw'
  left_info_topic   = camera + '/left/camera_info'
  right_image_topic = camera + '/right/image_raw'
  right_info_topic  = camera + '/right/camera_info'
  topics = [left_image_topic, left_info_topic, right_image_topic, right_info_topic]
  
  # Prepare the output file to store the odometry msgs
  rostopic_cmd = ['rostopic','echo','-p','/mosaic_processor/odom_gt']
  print '=== running rostopic:',' '.join(rostopic_cmd)
  logfile = open(output, 'w')
  rostopic_process = subprocess.Popen(rostopic_cmd, stdout=logfile)
  #time.sleep(2)

  image_proc_cmd = ['rosrun', 'stereo_image_proc', 'stereo_image_proc','__ns:='+camera]
  print '=== running stereo_img_proc:',' '.join(image_proc_cmd)
  image_proc_process = subprocess.Popen(image_proc_cmd)
  
  rosparam_cmd = ['rosparam', 'load', param_file, 'mosaic_processor']
  print '=== loading parameters YAMLfile:',' '.join(rosparam_cmd)
  rosparam_process = subprocess.check_call(rosparam_cmd)
  
  mosaic_cmd = ['rosrun', 'mosaic_pose_extractor', 'mosaic_processor', 'stereo:=/stereo_down', 'image:=image_rect_color']
  print '=== running mosaic_processor:',' '.join(mosaic_cmd)
  mosaic_process = subprocess.Popen(mosaic_cmd)
  
  # slow playback to have more point clouds from stereo_image_proc
  bag_play_cmd = ['rosbag', 'play', '--clock', '-r', '0.1', '-d', '5.0']
  for bag in inbags:
    bag_play_cmd.append(bag)
  bag_play_cmd.append('--topics')
  for t in topics:
    bag_play_cmd.append(t)
  print '=== running bagfile:',' '.join(bag_play_cmd)
  play_process = subprocess.Popen(bag_play_cmd)

  play_process.wait()
  # stop processes when playback finished
  image_proc_process.send_signal(signal.SIGINT)
  image_proc_process.wait()
  mosaic_process.send_signal(signal.SIGINT)
  mosaic_process.wait()
  rostopic_process.send_signal(signal.SIGINT)
  rostopic_process.wait()
  print '=== bagfile ended.'
  return output

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Obtain odometry ground truth from a bagfile and save it in a file.')
  parser.add_argument('inbag', help='input bagfile', nargs='+')
  parser.add_argument('-t', '--start_time', type=float, help='Start time in the bagfile. Default = 0.0')
  parser.add_argument('-d', '--duration', type=float, help='Time window')
  parser.add_argument('-o', '--output', default='/home/miquel/output.dat', help='name of the output file')
  parser.add_argument('-p', '--params', default=roslib.packages.get_pkg_dir('mosaic_pose_extractor') + '/default_params.yaml', help='parameters YAMLfile to load')
  parser.add_argument('-c', '--camera', default='/stereo_down', help='base topic of the camera to use')
  args = parser.parse_args()
 
  try:
    getGroundTruth(args.inbag, args.start_time, args.duration, args.output, args.camera, args.params)
  except Exception, e:
    import traceback
    traceback.print_exc()
