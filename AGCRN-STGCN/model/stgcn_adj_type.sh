nohup python Run.py --model STGCN --adj_type I --use_ln True --device cuda:0 > PEMSD4-adj_type_I.out 2>&1 &
nohup python Run.py --model STGCN --adj_type mean --use_ln True --device cuda:0 > PEMSD4-adj_type_mean.out 2>&1 &
nohup python Run.py --model STGCN --adj_type I+mean --use_ln True --device cuda:1 > PEMSD4-adj_type_I+mean.out 2>&1 &
nohup python Run.py --model STGCN --adj_type rand --use_ln True --device cuda:1 > PEMSD4-adj_type_rand.out 2>&1 &
nohup python Run.py --model STGCN --adj_type I+rand --use_ln True --device cuda:1 > PEMSD4-adj_type_I+rand.out 2>&1 &
nohup python Run.py --model STGCN --adj_type None --use_ln True --device cuda:2 > PEMSD4-adj_type_origin.out 2>&1 &
nohup python Run.py --model STGCN --adj_type None --device cuda:2 > PEMSD4-adj_type_origin_without_ln.out 2>&1 &
nohup python Run.py --model STGCN --adj_type I --device cuda:2 > PEMSD4-adj_type_I_without_ln.out 2>&1 &


nohup python Run.py --model STGCN --graph_conv_type origin --device cuda:0 > PEMSD4-origin.out 2>&1 &
nohup python Run.py --model STGCN --graph_conv_type Mean --device cuda:0 > PEMSD4-Mean.out 2>&1 &
nohup python Run.py --model STGCN --graph_conv_type MeanP --device cuda:0 > PEMSD4-MeanP.out 2>&1 &
nohup python Run.py --model STGCN --graph_conv_type NoLNP --device cuda:1 > PEMSD4-NoLNP.out 2>&1 &
nohup python Run.py --model STGCN --graph_conv_type NoRes --device cuda:1 > PEMSD4-NoRes.out 2>&1 &
nohup python Run.py --model STGCN --graph_conv_type LNNoP --device cuda:1 > PEMSD4-LNNoP.out 2>&1 &