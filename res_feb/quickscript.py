for shape in [(5,40,25),(5,50,30),(5,50,40),(10,40,35),(2,100,40)]:
	config = f'N{shape[0]}_M{shape[1]}_K{shape[2]}'
	#print("\nConfiguration: ", config,'\n')
	cbss_c_conf = 0
	cbss_ch_conf = 0
	for i in range(1,51):
		with open(f'random-32-32-10/random-32-32-10_{config}_{i}_h0.txt','r') as file:
			for line in file:
				if 'num_conflicts' in line:
					conf = int(line.split(' ')[1])
	#				print(conf)
					cbss_c_conf += conf
		with open(f'random-32-32-10/random-32-32-10_{config}_{i}_h1.txt','r') as file:
			for line in file:
				if 'num_conflicts' in line:
					conf = int(line.split(' ')[1])
	#				print(conf)
					cbss_ch_conf += conf
	#print("Total conflicts in CBSS-c and CBSS-ch =", cbss_c_conf/50, '&', cbss_ch_conf/50)
	print('&',cbss_c_conf/50, '&', cbss_ch_conf/50)
