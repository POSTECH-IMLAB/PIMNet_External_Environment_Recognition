mkdir ../../data/mot17
cd ../../data/mot17
wget https://motchallenge.net/data/MOT17.zip



unzip MOT17.zip

rm MOT17.zip

mkdir annotations
cd ../../src/tools/
python convert_mot_to_coco.py
python convert_mot_det_to_results


mkdir ../../data/mot20
cd ../../data/mot20
wget https://motchallenge.net/data/MOT20.zip



unzip MOT20.zip

rm MOT20.zip

cd ../../src/tools/


