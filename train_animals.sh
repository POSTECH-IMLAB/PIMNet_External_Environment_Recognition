# Backup
#./darknet detector train cfg/animals.data cfg/animals.cfg backup/animals.backup -gpus 0,1
# init_train
./darknet detector train cfg/animals.data cfg/animals.cfg weights/darknet19_448.conv.23 -gpus 0,1

# test22
