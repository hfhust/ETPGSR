CUDA_VISIBLE_DEVICES="3" python3 main.py --arch="tsrn_tl_cascade" --batch_size=48 --STN --mask --use_distill --gradient --sr_share --stu_iter=1 --vis_dir='vis_TPGSR-TSRN-lbo-0507bk'
