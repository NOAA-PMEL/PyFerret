	SUBROUTINE MEM_TO_BYTES (mem, mem_str, slen, mem_unit)


* Convert memory in Words to bytes, making it Kb, Mb or Gb depending on size.
* Returns two strings with the size and the units. slen is the length of mem_str

* NOAA/PMEL, Seattle, WA - Science Data Integration Group


* V751  6/1  *acm*

	INTEGER*8 mem
	INTEGER   slen
	CHARACTER*(*) mem_str, mem_unit

	CHARACTER TM_FMT_TRIM*64

	IF (mem .GT. 100000000) THEN
           mem_str = TM_FMT_TRIM( FLOAT(mem*8)/1.E9, -2, 12, slen)
           mem_unit = 'Gb'
        ELSE IF (mem .GT. 100000) THEN 
           mem_str = TM_FMT_TRIM( FLOAT(mem*8)/1.E6, -2, 12, slen)
           mem_unit = 'Mb'
        ELSE 
           mem_str = TM_FMT_TRIM( FLOAT(mem*8)/1.E3, -2, 12, slen)
           mem_unit = 'Kb'
        ENDIF

	RETURN
	END

