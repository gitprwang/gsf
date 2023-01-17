echo 'STGCN#'
python time_test.py --model STGCN --origin True --use_ln True
echo 'STGCN'
python time_test.py --model STGCN
echo 'AGCRN#'
python time_test.py --model AGCRN --origin True
echo 'AGCRN'
python time_test.py --model AGCRN
