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



static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(toggleButton62),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_corners_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_corners_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_corners_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton63),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton64),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton37),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpfull_xpm),
		XmNselectPixmap, GetPixmapFromData(vpfull_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton38),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpll_xpm),
		XmNselectPixmap, GetPixmapFromData(vpll_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton39),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplr_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplr_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);  

	XtVaSetValues(UxGetWidget(toggleButton40),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpur_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpur_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton41),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpll_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpll_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton42),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpleft_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpleft_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton43),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpright_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpright_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton44),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpupper_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpupper_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton45),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplower_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplower_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);
}

static void ClearBtnCB()
{
	ferret_command("SET WINDOW/CLEAR", IGNORE_COMMAND_WIDGET);
	/*gSomethingIsPlotted = 0;
	MaintainMainWdBtns(); */
}

static void InitInterface()
{
	/* initialize the viewports */
	ClearAllViewPorts();

	/* cycle buttons insensitive */
	ClearCycleButtons();

	if (gViewportActive) {
		switch (gCurrViewportType) {
			case 0:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton38),
							XmNset, True,
							NULL);
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton39),
							XmNset, True,
							NULL);
						break;
					case 2:
						XtVaSetValues(UxGetWidget(toggleButton40),
							XmNset, True,
							NULL);
						break;
					case 3:
						XtVaSetValues(UxGetWidget(toggleButton41),
							XmNset, True,
							NULL);
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton62),
						XmNset, True,
						NULL);
				}
				break;
			case 1:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton42),
							XmNset, True,
							NULL);
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton43),
							XmNset, True,
							NULL);
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton63),
						XmNset, True,
						NULL);
				}
				break;
			case 2:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton44),
							XmNset, True,
							NULL);
						break;
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton45),
							XmNset, True,
							NULL);
						break;
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton64),
						XmNset, True,
						NULL);
				}
				break;
		}
	}
	else {
		/* full window selected */
		XtVaSetValues(UxGetWidget(toggleButton37),
			XmNset, True,
			NULL);
	}
}

static void ClearCycleButtons()
{
	XtVaSetValues(UxGetWidget(toggleButton62),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton63),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton64),
		XmNset, False,
		NULL);
}

static void ClearAllViewPorts()
{
	XtVaSetValues(UxGetWidget(toggleButton37),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton38),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton39),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton40),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton41),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton42),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton43),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton44),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton45),
		XmNset, False,
		NULL);
}
