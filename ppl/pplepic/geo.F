	FUNCTION GEO (SLAT,NSLAT)
C** 
C**    @(#)geo.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
C
C  RECEIVES CHARACTER LATITUDE OR LONGITUDE (D[DD MM.M]S) AND RETURNS 
C  REAL NUMBER LATITUDE OR LONGITUDE IN DEGREES (+ FOR NORTH, - FOR SOUTH)
C  (+ FOR WEST, 360-LONGITUDE FOR EAST)
C
C  PROGRAMMED BY N SOREIDE, SEP 84
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER SLAT*(*), HEMI*1
C	CHARACTER HOLD*15
	HOLD = 110
	RM = 0.
	IF (NSLAT.EQ.1 .AND. SLAT(1:1).EQ.'0') THEN
		GEO=0.
	ELSE
		HEMI = SLAT(NSLAT:nslat)
		read(slat(1:nslat-1),*)RD, RM
		GEO = RD + RM/60.
		IF (HEMI .EQ. 'S') GEO = -GEO
		IF (HEMI .EQ. 'E') GEO = 360.-GEO
	ENDIF
	RETURN
	END
