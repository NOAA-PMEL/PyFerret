/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/



/*---------------------------------------------------------------------
 * $Date$             $Revision$
 *---------------------------------------------------------------------
 * 
 *
 *             Copyright (c) 1992, Visual Edge Software Ltd.
 *
 * ALL  RIGHTS  RESERVED.  Permission  to  use,  copy,  modify,  and
 * distribute  this  software  and its documentation for any purpose
 * and  without  fee  is  hereby  granted,  provided  that the above
 * copyright  notice  appear  in  all  copies  and  that  both  that
 * copyright  notice and this permission notice appear in supporting
 * documentation,  and that  the name of Visual Edge Software not be
 * used  in advertising  or publicity  pertaining to distribution of
 * the software without specific, written prior permission. The year
 * included in the notice is the year of the creation of the work.
 *-------------------------------------------------------------------*/
/*------------------------------------------------------------------------
 * Method registration and dispatch for swidget interfaces.
 * This version supports the default macros and generated code,
 * using a simple (and fast) table lookup approach.
 *------------------------------------------------------------------------*/

#ifdef XT_CODE
#	include "UxXt.h"
#else /* XT_CODE */
#	include "UxLib.h"
#	include "method.h"
#endif /* XT_CODE */


#ifdef DESIGN_TIME
/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include "veos.h"
#include "swidget.h"
#endif

/* Global environment variable supplied for the user's convenience */
Environment UxEnv = {NO_EXCEPTION};

static char **Names;		/* All method names as registered */
static int   NameLen = 0;	/* Gross length of names array	*/	
static int   NumNames = 0;	/* Num Elements in names array	*/

static void  ***MethodTable;	/* All method pointers 		*/
static int   *BaseTable;	/* Base class id's by class id. */
static int   NumClasses = 0;	/* Active width of method table	*/
static int   *MethodCounts;	/* Height by row in method table */

#ifdef _NO_PROTO
static int UndefinedMethod ();
#else /* _NO_PROTO */
static int UndefinedMethod (void);
#endif /* _NO_PROTO */

#ifdef XT_CODE
static  XContext        xcontext_mid = 0;
#endif /* XT_CODE */

/*------------------------------------------------------------------------
 * NAME: UxMethodLookup
 * INPUT:	swidget sw;	-- a swidget, usually an interface swidget
 *		int	mid;	-- a method id from UxMethodRegister
 * RETURNS:	Pointer to method function for this method.
 * DESCRIPTION:
 *	This is the heart of virtual method lookup.
 *	The design-time routing functions and runtime macros
 *	use this function to fnd the implementation of a method.
 *	The return value is the actual function pointer
 *	for the implementation of the method associated with `sw's 
 *	interface class.
 *	
 * LAST REV:	Feb 93	fix3910		Add support for subclassing.
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
void* UxMethodLookup(sw, mid, mname) 
	swidget sw;
	int	mid;
	char	* mname;
#else /* _NO_PROTO */
void* UxMethodLookup(swidget sw, int mid, char	*mname)
#endif /* _NO_PROTO */
{
	int cid = UxGetClassCode(sw);

	if (mid < 0) {
		mid = UxMessageIndex(mname);
		if (mid < 0) {
			return 0;
		}
	} 

	/*-----------------------------------------------------------
	 * The method may be registered on this class or a base class.
	 *-----------------------------------------------------------*/

	while (cid > -1 && cid < NumClasses)
	{
		if (mid < MethodCounts[cid] && MethodTable[cid][mid]) 
		{
			return MethodTable[cid][mid];
		}
		cid = BaseTable[cid];
	}
#ifndef DESIGN_TIME
	return (void *) UndefinedMethod;
#else 
	return 0;
#endif 
}

/*------------------------------------------------------------------------
 * NAME: UxNewClassId 
 * RETURNS:	a unique integer each time called.
 * DESCRIPTION:
 *	This is to get a unique id for each interface class registered.
 *	The result is used for calls to UxMethodRegister, and
 *	is associated with each instance of the interface class:
 *	it is returned by UxGetClassId(sw) for each interface swidget sw
 *	that is an instance of the interface it represents.
 *
 *	We have no subclassing.  When it comes, parent class id's
 *	will be supplied in this call, and used to initialize the 
 *	classes' method table with its parent's versions.
 *
 * LAST REV:	Feb 93	fix3910		Add support for subclassing.
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
int UxNewClassId()
#else /* _NO_PROTO */
int UxNewClassId(void)
#endif /* _NO_PROTO */
{
	static int   GrossClasses = 0;	/* Gross width of method table  */

	int cid = NumClasses++;

	if (GrossClasses < NumClasses)
	{
		if (GrossClasses) {
			MethodTable = (void***) 
			        UxRealloc(MethodTable, 
				       (GrossClasses += 10) * sizeof(void**));
			MethodCounts = (int*) 
				UxRealloc(MethodCounts,
					  (GrossClasses * sizeof(int)));
			BaseTable = (int*) 
				UxRealloc(BaseTable,
					  GrossClasses * sizeof(int));
		} else {
			MethodTable = (void***) UxCalloc((GrossClasses += 10), 
					       		 sizeof(void**));
			MethodCounts = (int*) UxCalloc(GrossClasses, 
						       sizeof(int));
			BaseTable = (int*) UxCalloc(GrossClasses, 
						       sizeof(int));
		}
	}
	MethodTable[cid]   = (void**) UxCalloc(NumNames, sizeof(void*));
	MethodCounts[cid]  = NumNames;
	BaseTable[cid]  = -1;

	return cid;
}

/*------------------------------------------------------------------------
 * NAME: UxNewSubclassId (super)
 * 	<register a new class as a subclass of an existing class> 
 * INPUT:	super class id
 * RETURNS:	new class id of a subclass of the superclass.
 *
 * LAST REV:	Feb 93	fix3910		Add support for subclassing.
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
int 	UxNewSubclassId(super)
	int super;
#else /* _NO_PROTO */
int 	UxNewSubclassId( int super)
#endif /* _NO_PROTO */
{
	int cid = UxNewClassId();
	BaseTable[cid] = super;
	return cid;
}

/*------------------------------------------------------------------------
 * NAME:	UxMessageIndex 
 * INPUT:	char* name; -- name of a method being registered.
 * RETURNS:	A unique id number for this name.
 * DESCRIPTION:
 *	Assigns a unique integer to a message name.  
 *	Must return the same id for all calls with the same name.
 *
 *	This is used while rgistering methods at startup time.
 *	The result is used when method calls are made
 *	(see UxMethodLookup).  If a very large number of method names
 *	are being used, initialization could be speeded up by making
 *	this function faster (by sorting or hashing the names).
 *
 * LAST REV:	June 92	fix3574		Created.
 *------------------------------------------------------------------------*/
#ifdef _NO_PROTO
int UxMessageIndex (name)
	char* name;
#else /* _NO_PROTO */
int UxMessageIndex ( char* name)
#endif /* _NO_PROTO */
{
	int mid;

	for (mid = 0 ; mid < NumNames; mid++)
	{
		if (UxStrEqual(Names[mid], name)) {
			return mid;
		}
	}
	if (mid >= NameLen) {
		if (NameLen == 0) {
			Names = (char**) UxMalloc((NameLen=64)*sizeof(char*));
		} else {
			Names = (char**) UxRealloc(Names, 
						   (NameLen*=2)*sizeof(char*));
		}
	}

	Names[NumNames] = (char*) UxMalloc(strlen(name)+1);
	strcpy(Names[NumNames], name);
	return NumNames++;
}

/*------------------------------------------------------------------------
 * NAME: MethodTableSpace 
 * INPUT:
 *		int cid;	-- class id from UxNewClassId ()
 *		int mid;	-- method id  from UxMessageIndex ()
 * DESCRIPTION:
 *
 *  Enlarges the method table for the given cid,
 *  so that it is at least big enough to register a method
 *  for the given message index.  We also add a little more space
 *  (two more cells), but not a lot: there's one of these
 *  arrays for each class, so we don't want them growing too large.
 *
 * LAST REV:	Feb 93	fix3910		Add support for subclassing.
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
static void MethodTableSpace (cid, mid)
	int cid;
	int mid;
#else /* _NO_PROTO */
static void MethodTableSpace (int cid, int mid)
#endif /* _NO_PROTO */
{
	int old_cnt = MethodCounts[cid];
	int i;

	if (old_cnt < mid+1)
	{
		MethodTable[cid] = (void**) UxRealloc(MethodTable[cid],
						     (mid + 2) * sizeof(void*));
		for (i = old_cnt; i < mid+2; i++)
		{
			MethodTable[cid][i] = 0;
		}
		MethodCounts[cid] = mid + 2;
	}
}

/*------------------------------------------------------------------------
 * NAME: UxMethodRegister
 * INPUT: 	int 	cid;		-- a class code from UxNewClassId()
 *		char 	*name;		-- a method name
 *		void 	(*function) ();	-- implementation of the method
 *					   for the identified class.
 *
 * RETURNS:	An identifier code for the given method name
 *		to be used in subsequent calls to UxMethodLookup.
 *		All calls to this function with the same method name
 *		return the same index.
 *
 * DESCRIPTION:
 *	Called during interface class initialization for each method
 *	implemented by the class.  
 *	It builds the method lookup table for the class. 
 *
 * LAST REV:	June 92	fix3574		Created.
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
int UxMethodRegister(cid, name, function)
	int 	cid;
	char 	*name;
	void 	(*function) ();
#else /* _NO_PROTO */
int UxMethodRegister(int cid, char *name, void 	(*function) ())
#endif /* _NO_PROTO */
{
	int mid = UxMessageIndex (name);

	MethodTableSpace(cid, mid);
	MethodTable[cid][mid] = (void *) function;

	return mid;
}

/*------------------------------------------------------------------------
 * NAME:  UxGetClassCode
 *
 * INPUT:	some swidget
 * RETURNS:		its ifClassCode, or that of its nearest container
 *			that has a class code.
 * DESCRIPTION:
 *
 *	This makes calling methods a little more flexible.
 *	Class codes are assigned to top-level swidgets in interfaces,
 *	when their delarations are initialized. 
 *
 * LAST REV:
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
int	UxGetClassCode(sw)
	swidget sw;
#else /* _NO_PROTO */
int	UxGetClassCode(swidget sw)
#endif /* _NO_PROTO */
{
	while (sw && UxGetIfClassCode(sw) < 0) {
		sw = UxGetParent(sw);
	}

	if (sw)
		return UxGetIfClassCode(sw);

	return -1;
}

/******************************************************************************
NAME:		UxPutClassCode( wgt, id )

INPUT:		Widget	wgt		- Widget
		int	id;		- Id od method

RETURN:		int			UX_ERROR / UX_NO_ERROR

DESCRIPTION:	Uses the X Context manager to store the given id 
		in a memory location that is indexed by the given widget id.

EXT REFERENCES:	UxTopLevel, xcontext_mid
EXT EFFECTS:	xcontext_mid

CREATION:	Visual Edge Software		January 9 1993
-----------------------------------------------------------------------------*/
#ifdef XT_CODE
#ifdef _NO_PROTO
int	UxPutClassCode( wgt, id)
	Widget		wgt;
	int		id;
#else
int	UxPutClassCode( Widget wgt, int id)
#endif /* _NO_PROTO */
{
	int		status;

	if ( xcontext_mid == 0 )
		xcontext_mid = XUniqueContext();

	if ( wgt == NULL )
		return ( UX_ERROR );

	status = XSaveContext( XtDisplay( UxTopLevel ), 
			       (Window) wgt, 
			       xcontext_mid, 
			       (XtPointer) id );
	if ( status != 0 )
		return ( UX_ERROR );

	XtAddCallback (wgt, XmNdestroyCallback, 
		UxDeleteContextCB, (XtPointer) xcontext_mid);

	return ( UX_NO_ERROR );
}
#endif /* XT_CODE */

/******************************************************************************
NAME:		UxGetIfClassCode( wgt )

INPUT:		Widget	wgt		- widget

RETURN:		caddr_t			- the context pointer

DESCRIPTION:	Uses the X Context manager to find the method id 
		stored in a memory location indexed by the given widget id.

EXT REFERENCES:	UxTopLevel, xcontext_mid

CREATION:	Visual Edge Software		January 9 1993
-----------------------------------------------------------------------------*/
#ifdef XT_CODE
#ifdef _NO_PROTO
int		UxGetIfClassCode( wgt )
		Widget	wgt;
#else
int		UxGetIfClassCode( Widget wgt )
#endif /* _NO_PROTO */
{
	int		status;
	XtPointer	id;

	if ( wgt == NULL )
		return -1;

	status = XFindContext( XtDisplay( UxTopLevel ), 
			       (Window) wgt, 
			       xcontext_mid, 
			       (XtPointer) &id );

	if ( status != 0 )
		return	-1;

	return ( (int) id);
}
#endif /* XT_CODE */

/*------------------------------------------------------------------------
 * UxChildSite (this)
 * INPUT:	swidget inst - user-defined component
 * Gets the designatedChildSite by composing a call to the childSite
 * method and calling that method. If the childSite is an intance,
 * we recur by calling the childSite's childSite method and so on...
 * LAST REV:	March 1993	3988	- don't recur when childSite==this
 *------------------------------------------------------------------------*/
#ifdef _NO_PROTO
swidget	UxChildSite (inst)
	swidget inst;
#else /* _NO_PROTO */
swidget	UxChildSite (swidget inst)
#endif /* _NO_PROTO */
{
	/*-----------------------------------------------------
	 * Index of "childSite" in low-level method dispatcher.
	 *-----------------------------------------------------*/
	static int	ChildSiteMID = -1;

	swidget childSite 	= NULL;
#ifdef _NO_PROTO
	void    (*childSiteMethod) () ;
#else
	void    (*childSiteMethod) (swidget, Environment *);
#endif

	if (ChildSiteMID == -1) {
		ChildSiteMID = UxMessageIndex("childSite");
	}
	if (! inst) {
		return childSite;
	}

	childSiteMethod = (void (*) ()) 
			  UxMethodLookup (inst, ChildSiteMID, "childSite");
	if (childSiteMethod) {
		childSite = ((swidget(*)())childSiteMethod) (inst, &UxEnv);
		if (childSite && childSite != inst) 
		{
			swidget nested = UxChildSite(childSite);
			if (nested) {
				childSite = nested;
			}
		}
	}
	if (!childSite) {
		childSite = inst;
	}
	return childSite;
}

/*------------------------------------------------------------------------
 * NAME: UxReclassifyCID (cid, super)
 * 	<redefine a subclass as being derived from a new base.>
 * INPUT:	sub class id
 *		super class id
 *
 * LAST REV:	3988 Mar 93	Created.
 *------------------------------------------------------------------------*/
#ifdef DESIGN_TIME

#ifdef _NO_PROTO
void 	UxReclassifyCID(cid, super)
	int cid;
	int super;
#else /* _NO_PROTO */
void 	UxReclassifyCID(int cid, int super)
#endif /* _NO_PROTO */
{
	if (cid == super) {
		UxInternalError(__FILE__, __LINE__, "Cid == Super Cid");
	}
	BaseTable[cid] = super;
}
#endif  /* DESIGN_TIME */
/*------------------------------------------------------------------------
 * NAME: UxInheritedMethodLookup
 * INPUT:	swidget sw;	-- a swidget, usually an interface swidget
 *		int	mid;	-- a method id from UxMethodRegister
 * RETURNS:	Pointer to method function for this method.
 * DESCRIPTION:
 *	Like UxMethodLookup, but starts searching at the first
 *	base class of the given sw, not at its own class.
 *	
 * LAST REV:	3988 March 93 Created.
 *------------------------------------------------------------------------*/
#ifdef DESIGN_TIME

#ifdef _NO_PROTO
void*	UxInheritedMethodLookup (sw, mid, mname)
	swidget sw;
	int	mid;
	char	* mname;
#else /* _NO_PROTO */
void*	UxInheritedMethodLookup (swidget sw, int mid, char *mname)
#endif /* _NO_PROTO */
{
	int cid = UxGetClassCode(sw);

	if (cid < 0 ||  cid > NumClasses) {
		return 0;
	}

	if (mid < 0) {
		mid = UxMessageIndex(mname);
		if (mid < 0) {
			return 0;
		}
	} 

	cid = BaseTable[cid];
	while (cid > -1 && cid < NumClasses)
	{
		if (mid < MethodCounts[cid] && MethodTable[cid][mid]) {
			return MethodTable[cid][mid];
		}
		cid = BaseTable[cid];
	}
	return 0;
}
#endif  /* DESIGN_TIME */

#ifndef DESIGN_TIME
/*------------------------------------------------------------------------
 * NAME: UndefinedMethod
 * INPUT:	
 * RETURNS:	always 0.
 * DESCRIPTION: Function used as a return value of UxMethodLookup
 *		in case the method lookup fails.
 *	
 * LAST REV:	April 8 1993
 *------------------------------------------------------------------------*/

#ifdef _NO_PROTO
static int UndefinedMethod ()
#else /* _NO_PROTO */
static int UndefinedMethod (void)
#endif /* _NO_PROTO */
{
	return 0;
}
#endif /* DESIGN_TIME */
