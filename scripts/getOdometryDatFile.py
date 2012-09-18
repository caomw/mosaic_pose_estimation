#!/usr/bin/python

PKG = 'mosaic_cam_pose' # this package name

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
 
def getGroundTruth(inbags, start_time, duration, output, camera):
  # cut_bag_name = cut_bag(inbags, start_time, duration)
  left_image_topic  = camera + '/left/image_raw'
  left_info_topic   = camera + '/left/camera_info'
  right_image_topic = camera + '/right/image_raw'
  right_info_topic  = camera + '/right/camera_info'
  topics = [left_image_topic, left_info_topic, right_image_topic, right_info_topic]
  
  # Prepare the output file to store the odometry msgs
  rostopic_cmd = ['rostopic','echo','-p','/odom_gt']
  print '=== running rostopic:',' '.join(rostopic_cmd)
  logfile = open(output, 'w')
  rostopic_process = subprocess.Popen(rostopic_cmd, stdout=logfile)
  #time.sleep(2)

  image_proc_cmd = ['rosrun', 'stereo_image_proc', 'stereo_image_proc','__ns:='+camera]
  print '=== running stereo_img_proc:',' '.join(image_proc_cmd)
  image_proc_process = subprocess.Popen(image_proc_cmd)
  
  mosaic_cmd = ['rosrun', 'mosaic_cam_pose', 'mosaic_processor', 'stereo:=/stereo_down', 'image:=image_rect_color']
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
  parser.add_argument('-c', '--camera', default='/stereo_down', help='base topic of the camera to use')
  args = parser.parse_args()
 
  try:
    getGroundTruth(args.inbag, args.start_time, args.duration, args.output, args.camera)
  except Exception, e:
    import traceback
    traceback.print_exc()
