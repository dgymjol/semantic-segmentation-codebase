# for debugging
# CUDA_VISIBLE_DEVICES=1 python train.py


# # test 2

# cd experiment/deeplabv3_ddp_irn_bdcl_test2
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py
# sleep 20
# python test.py
# sleep 20
# cd ../../
# mv log/logfile.txt log/deeplabv3_ddp_irn_bdcl_test2/logfile.txt

# sleep 10
# git add .
# sleep 10
# git commit -m "exp 2"
# sleep 10
# git push
# sleep 10

# # test 3

# cd experiment/deeplabv3_ddp_irn_bdcl_test3
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py
# sleep 20
# python test.py
# sleep 20
# cd ../../
# mv log/logfile.txt log/deeplabv3_ddp_irn_bdcl_test3/logfile.txt

# sleep 10
# git add .
# sleep 10
# git commit -m "exp 3"
# sleep 10
# git push
# sleep 10

# # test 4

# cd experiment/deeplabv3_ddp_irn_bdcl_test4
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py
# sleep 20
# python test.py
# sleep 20
# cd ../../
# mv log/logfile.txt log/deeplabv3_ddp_irn_bdcl_test4/logfile.txt

# sleep 10
# git add .
# sleep 10
# git commit -m "exp 4"
# sleep 10
# git push
# sleep 10





# test 2

cd experiment/deeplabv3_ddp_irn_bdcl_test2
python3 -m torch.distributed.launch --nproc_per_node=2 train.py
sleep 20
python test.py
sleep 20
cd ../../
mv log/logfile.txt log/deeplabv3_ddp_irn_bdcl_test2/logfile.txt

sleep 10
git add .
sleep 10
git commit -m "exp 2"
sleep 10
git push
sleep 10

# test 3

cd experiment/deeplabv3_ddp_irn_bdcl_test3
python3 -m torch.distributed.launch --nproc_per_node=2 train.py
sleep 20
python test.py
sleep 20
cd ../../
mv log/logfile.txt log/deeplabv3_ddp_irn_bdcl_test3/logfile.txt

sleep 10
git add .
sleep 10
git commit -m "exp 3"
sleep 10
git push
sleep 10


