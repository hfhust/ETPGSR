CUDA_VISIBLE_DEVICES="1" python3 main.py --arch="tsrn_tl_cascade" --test_model="CRNN" --batch_size=48 --STN --mask  --sr_share --gradient --go_test --stu_iter=1 --vis_dir='vis_TPGSR-TSRN-lbo-0507bk' 
