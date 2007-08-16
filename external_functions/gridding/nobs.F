

c************************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy=x/y location of data pt (data units)
c  i	nx,ny=size of grids
c  i	x1,y1=west/south edge of grid (center of 1st box in data units)
c  o 	obs(nx,ny) number of obs in each (x,y) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobs (xx, yy, nx, ny, x1, y1, dx, dy, 
     .			 xf, yf, obs)


* Argument declarations

        INTEGER nx, ny
	REAL xx, yy, dx, dy, x1, y1, xf, yf, obs(nx,ny)

* Local variable declarations

        REAL xxg, yyg
        INTEGER ig, jg

	xxg = (xx-x1)/ dx + 1.		  ! grid values of data location
	yyg = (yy-y1)/ dy + 1.

c.................save the number of obs in each (x,y) gridbox
	ig = NINT(xxg)
	jg = NINT(yyg)
	IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny)
     .		obs(ig,jg) = obs(ig,jg)+1.

	RETURN
	END


