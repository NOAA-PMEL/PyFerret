#ifndef _FERRET_H 
#define _FERRET_H


/* non-ANSI function prototypes */
#ifdef NO_ENTRY_NAME_UNDERSCORES      /*added ifdef for HP port *kob* 6.95*/
void ctrlc_ast();   /* pointer to ^C interrupt routine */
void initialize();
void set_ctrl_c(); /* void set_ctrl_c_( void (*CTRLC_AST_)() ); */ 
void help_text();
void init_journal( );
void proclaim_c( );
void init_memory( );
void ferret_dispatch( );
void save_ppl_memory_size( );
void get_ppl_memory_size( );
void reallo_ppl_memory();
/* 04.20.99 *jc*
 * Folded in from fer/gui/ferret_fortran.h
 */
void ferret_dispatch_c();
void WindowMapping();
void mode_gui_on();
void secs_to_date_c();
double tm_secs_from_bc();
void xgks_x_events();
void gescinqxattr();
/*
 * End of 04.20.99
 */
#else
void ctrlc_ast_();   /* pointer to ^C interrupt routine */
void initialize_();
void set_ctrl_c_(); /* void set_ctrl_c_( void (*CTRLC_AST_)() ); */ 
void help_text();
void init_journal_( );
void proclaim_c_( );
void init_memory_( );
void ferret_dispatch_( );

/* new 10/01  for PPLUS dynamic memory*/
void save_ppl_memory_size_( );
void get_ppl_memory_size_( );
/* 04.20.99 *jc*
 * Folded in from fer/gui/ferret_fortran.h
 */
void ferret_dispatch_c();
void WindowMapping();
void mode_gui_on();
void secs_to_date_c();
double tm_secs_from_bc_();
void xgks_x_events();
void gescinqxattr();
/*
 * End of 04.20.99
 */

#endif

/* memory configuration defaults */
/* NOTE!! PMEM_BLK_SIZE must match pmem_blk_size in xvariables.cmn */
#define PMEM_BLK_SIZE 2*(160*100*4/10)  /*  2* 9/01 *sh* */
#define PMAX_MEM_BLKS 500

/* from XPROG_STATE COMMON */
#define TTOUT_LUN 6

/* these parameters describe the first few integers of the buffer
 returned by FERRET to its GUI*/
#define			FRTN_CONTROL  0    /* 1 in FORTRAN */
#define			FRTN_STATUS   1    /* 2 in FORTRAN */
#define			FRTN_ACTION   2    /* 3 in FORTRAN */
#define			FRTN_IDATA1   5    /* 6 in FORTRAN */
#define			FRTN_IDATA2   6    /* 7 in FORTRAN */
#define			FRTN_IDATA3   7    /* 8 in FORTRAN */

/* who is in control according to return_buff(frtn_control) ?
when the GUI is running FERRET control may return to the GUI at times
other than at the completion of a command - for example, when FERRET
is requesting that a warning message be displayed or that memory be
reconfigured.  These codes indicate why FERRET has returned.
FERRET will reset the control variable to "ctrl_not_finished" if
the given command was really multiple commands and they are not yet complete*/
#define	                FCTRL_BACK_TO_GUI  1
#define			FCTRL_IN_FERRET    2

/* what special action has FERRET requested in return_buff(frtn_action) ? */
#define			FACTN_NO_ACTION		   0
#define			FACTN_MEM_RECONFIGURE  1
#define			FACTN_EXIT		       2
#define			FACTN_DISPLAY_WARNING  3
#define			FACTN_DISPLAY_ERROR	   4
#define			FACTN_DISPLAY_TEXT	   5
#define         FACTN_SYNCH_SET_DATA   6  /* added 11/1/94 */
#define         FACTN_SYNCH_LET        7
/* 04.20.99 *jc*
 * Folded in from fer/gui/ferret_fortran.h
 */
#define         FACTN_SYNCH_WINDOW     8
#define         FACTN_PAUSE            10
/*
#ifdef __globalDefs
#define __global
#else
#define __global extern
#endif

__global float *memory;

#undef global
*/
/*
 * End of 04.20.99
 */

/* Easier way of handling FORTRAN calls with underscore/no underscore */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif

#endif /* _FERRET_H */

