import subprocess
import sys

cmd1 = "cd /nfs/project/libo_i/mmdetection"
cmd2 = "python3 tools/test.py configs/{}/{}.py " \
       "work_dirs/{}_{}/epoch_{}.pth --gpus 4 --out val.pkl --eval bbox segm --proc_per_gpu 4"

config = sys.argv[1]
task = sys.argv[2]

for i in range(8, 13):
	cmd = cmd1 + " && " + cmd2.format(task, config, task, config, i, task, config)
	print(cmd)
	subprocess.call(cmd, shell=True)
