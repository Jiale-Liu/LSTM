def Read(start,end):
	f = open("combine.txt")
	array = []
	line = f.readline()
	point = 0;
	mi = 0;
	while line:
		point = point + 1
		if(point>=start and point<=end):
			strArr = line.split()
			name_emb = {'num1':float(strArr[0]),'num2':float(strArr[1]),'num3':float(strArr[2]),'num4':float(strArr[3])}
			array.append(name_emb)
		line = f.readline()
	f.close()
	return array


def msort(array):
	point = 0;
	array.sort(key = lambda x:x["num4"],reverse = True)
	for po in range(len(array)):
		if array[po]['num2']==1:
			point = po
			break
	print(point)
	return point




msArr = Read(1,73980)
msort(msArr)
msArr = Read(73981,111720)
msort(msArr)
msArr = Read(111721,335160)
msort(msArr)
msArr = Read(335161,488520)
msort(msArr)
msArr = Read(488521,542475)
msort(msArr)
msArr = Read(542476,585995)
msort(msArr)
msArr = Read(585996,688485)
msort(msArr)
msArr = Read(688486,1010115)
msort(msArr)
msArr = Read(1010116,1209615)
msort(msArr)
msArr = Read(1209616,1245015)
msort(msArr)
msArr = Read(1245016,1327815)
msort(msArr)
