	subroutine fakes_tmio( d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,
     .                         d11,d12,d13,d14,d15,status)

* arguments
	integer d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,
     .                         d11,d12,d13,d14,d15,status

**************
        entry	TM_INIT_DSET ( d1, d2, status)
        write (6,*)	'tm_init_dset has been removed for debugging'
        status = 0
        return
**************
        entry	TM_READ_GRIDFILE ( d1, status)
        write (6,*)	'tm_read_gridfile has been removed for debugging'
        status = 0
        return
**************
	entry    TM_READ( d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,
     .                         d11,d12,d13,d14,d15,status)
        write (6,*)	'tm_read has been removed for debugging'
        status = 0
        return

        end
