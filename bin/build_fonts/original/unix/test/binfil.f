	program binfil
	dimension x(63),y(63)
100	read(5,end=1000)x,y
	do 10 i=1,63
	write(6,999)x(i),y(i)
999	format(1x,1p2g12.4)
10	continue
	goto 100
1000	call exit
	end
