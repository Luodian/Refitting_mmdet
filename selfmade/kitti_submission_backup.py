if args.infer_submission:
	kitti_test_num = 200
	kitti_cls_num = 11
	pred_list_name = 'pred_list'
	pred_img_name = 'pred_img'
	
	test_image_path = "/nfs/project/libo_i/go_kitti/data/testing/image_2"
	test_image_list = os.listdir(test_image_path)
	
	pred_list_path = os.path.join(output_dir, pred_list_name)
	pred_img_path = os.path.join(output_dir, pred_img_name)
	
	# Ensure path exists.
	if not os.path.exists(pred_list_path):
		os.makedirs(pred_list_path)
	
	if not os.path.exists(pred_img_path):
		os.makedirs(pred_img_path)
	
	# assert kitti_test_num == len(test_image_list)
	
	for img_id in range(kitti_test_num):
		im = cv2.imread(os.path.join(test_image_path, test_image_list[img_id]))
		ist_cnt = 0
		text_save_name = "{}.txt".format(test_image_list[img_id][:-4])
		text_save_path = os.path.join(pred_list_path, text_save_name)
		file = open(text_save_path, "w")
		for cls_id in range(kitti_cls_num):
			if len(all_segms[cls_id]) != 0:
				cls_item = all_segms[cls_id][img_id]
				box_cls_item = all_boxes[cls_id][img_id]
				
				if len(cls_item) != 0:
					for ist_id, specific_item in enumerate(cls_item):
						# write image info
						box_score = box_cls_item[ist_id][4]
						if box_score < 0.5:
							continue
						mask = np.array(mask_util.decode(specific_item), dtype=np.float32)
						instances_graph = np.zeros((im.shape[0], im.shape[1]))
						instances_graph[mask == 1] = 255
						instance_save_name = "{}_{:0>3d}.png".format(
							test_image_list[img_id][:-4],
							ist_cnt)
						print(instance_save_name)
						instance_save_path = os.path.join(pred_img_path, instance_save_name)
						import scipy.misc as msc
						
						msc.imsave(instance_save_path, instances_graph)
						
						# write text info like that
						# ../pred_img/Kitti2015_000000_10_000.png 026 0.976347
						instance_info_To_Text = "../pred_img/{} {:0>3d} {}\n".format(instance_save_name,
						                                                             cls_id + 23,
						                                                             all_boxes[cls_id][img_id][
							                                                             ist_id][4])
						ist_cnt += 1
						file.writelines(instance_info_To_Text)
		
		file.close()
	exit(0)
