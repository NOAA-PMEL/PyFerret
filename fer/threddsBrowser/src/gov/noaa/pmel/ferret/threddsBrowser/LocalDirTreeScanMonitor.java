/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowser;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import javax.swing.JComponent;
import javax.swing.ProgressMonitor;

import thredds.catalog.InvCatalogImpl;

/**
 * A monitor that sets up and runs a LocalDirTreeScanner, displays the scan progress, and fires
 * off a PropertyChange event when the scan is complete.
 * 
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class LocalDirTreeScanMonitor extends PropertyChangeSupport implements PropertyChangeListener {
	private static final long serialVersionUID = 6099749370566968112L;

	/** Maximum length of a truncated path String */
	private final static int MAX_TRUNCATED_PATH_LENGTH = 60;

	/** root of the local directory tree to scan */
	File rootDir;
	/** flag so that the final property change event sent to the ThreddsBrowser is sent only once */
	boolean completed;
	/** scanner/catalog builder for the local directory tree */
	LocalDirTreeScanner scanner;
	/** dialog for displaying the progress of the scan */
	ProgressMonitor scanMonitor;

	/**
	 * Create a LocalDirTreeScanner as well as a ProgressMonitor.  The LocalDirTreeScanner will scan
	 * and build a catalog for the local directory tree rooted at localDir using datasetFilter as the
	 * file filter.  The ProgressMonitor will report the progress of this scan.  The scan is started
	 * (in the background) by calling the runScan method.  A PropertyChange event is fired off to all 
	 * registered PropertyChangeListeners when the scan is complete.  Possible values for this event are:
	 * <ul>
	 * <li> "Done": successful completion; getOldValue returns localDir, getNewValue returns the generated catalog
	 * <li> "Canceled": scan canceled or interrupted; getOldValue returns localDir, getNewValue returns null 
	 * 					(if canceled) or the InterruptedException (if interrupted, probably from being canceled)
	 * <li> "Died": scan threw an exception; getOldValue returns localDir, getNewValue returns the cause of
	 * 				the ExecutionException (ie, the exception thrown by the scanner) 
	 * </ul>
	 * @param parent the parent of the ProgressMonitor
	 * @param localDir the root of the local directory tree to scan
	 * @param datasetFilter the file filter for the scan
	 * @throws IOException if localDir is not a valid directory
	 */
	LocalDirTreeScanMonitor(JComponent parent, File localDir, FileFilter datasetFilter) throws IOException {
		super(new Object());
		rootDir = localDir;
		completed = false;
		// Create the actual scanner/catalog builder
		scanner = new LocalDirTreeScanner(rootDir, datasetFilter);
		// Have the scanner notify us when PropertyChange events are fired off
		scanner.addPropertyChangeListener(this);
		// Create a ProgressMonitor to go with the scanner (being a SwingWorker, progress values are [0,100])
		scanMonitor = new ProgressMonitor(parent, "Examining the local directory:                    ", 
										  truncatedPathname(rootDir, rootDir, MAX_TRUNCATED_PATH_LENGTH), 0, 100);
		/*
		 *  If taking more than 0.5 s, always popup the progress dialog.  Changed from defaults because the progress
		 *  estimate could easily be quite poor if there are lots of files in the last couple of directories.
		 */
		scanMonitor.setMillisToDecideToPopup(500);
		scanMonitor.setMillisToPopup(500);
	}

	/**
	 * Starts a background scan and build of the local directory tree catalog.
	 */
	public void runScan() {
		scanner.execute();
	}

	/**
	 * @return the total number of entries in the returned catalog.  If canceled,
	 * the total number of entries examined prior to cancellation.
	 */
	public int getNumCatalogEntries() {
		return scanner.getNumCatalogEntries();
	}

	/**
	 * This gets called when a property changes in the LocalDirTreeScanner 
	 * (a SwingWorker) or the ProgressMonitor associated with this object.
	 */
	@Override
	public void propertyChange(PropertyChangeEvent evt) {
		// If already fired off the final property change event just return
		if ( completed )
			return;

		// If the scanner has completed, fire off the final property change event
		if ( scanner.isDone() ) {
			completed = true;
			scanMonitor.close();
			if ( scanner.isCancelled() ) {
				firePropertyChange("Canceled", rootDir, null);
			}
			else {
				try {
					InvCatalogImpl catalog = scanner.get();
					firePropertyChange("Done", rootDir, catalog);
				} catch (InterruptedException e) {
					// scanner was interrupted (probably from a cancel)
					firePropertyChange("Canceled", rootDir, e);
				} catch (ExecutionException e) {
					// scanner threw an exception
					firePropertyChange("Died", rootDir, e.getCause());
				}
			}
			return;
		}

		/* 
		 * If the cancel button on the ProgressMonitor has been pressed, cancel the scan and let
		 * the resulting propertyChange event trigger the firing of the final property change event.
		 * This depends on the scanner firing off PropertyChange events often enough to quickly 
		 * detect this change.  Once the ProgressMonitor is canceled, isCanceled will always return 
		 * true and so all event handling will stop here.
		 */
		if ( scanMonitor.isCanceled() ) {
			scanner.cancel(true);
			return;
		}

		// If a progress update from the scanner, update the ProgressMonitor's progress bar
		if ( "progress".equals(evt.getPropertyName()) ) {
            int progress = (Integer) evt.getNewValue();
            scanMonitor.setProgress(progress);
            return;
        }

		// If an update in the directory being examined, update the ProgressMonitor's note
		if ( "Directory".equals(evt.getPropertyName()) ) {
			File dir = (File) evt.getNewValue();
			scanMonitor.setNote(truncatedPathname(dir, rootDir, MAX_TRUNCATED_PATH_LENGTH));
		}
	}

	/**
	 * Return a short pathname for the given File.  The name is truncated by using ellipses in
	 * the middle of the path to try keep the length from of the returned String from exceeding 
	 * MmaxLength.  If parent is given, the truncated name starts with the parent directory's 
	 * name if the given File is a directory under parent.
	 * @param fil the File to use
	 * @param parent the parent File to use as for a relative path (can be null)
	 * @param the maximum length of the returned short pathname
	 * @return the short pathname
	 */
	public static String truncatedPathname(File fil, File parent, int maxLength) {
		String filename = fil.getPath();
		// If parent is given and fil is under parent, always use the relative path
		if ( parent != null ) {
			String parentPath = parent.getPath();
			if ( filename.startsWith(parentPath) ) {
				String parentName = parent.getName();
				filename = parentName + filename.substring(parentPath.length());
			}
		}

		// If short enough, use the complete name
		if ( filename.length() <= maxLength )
			return filename;

		// Break up the path into each directory name components 
		// (a leading '/' will give an empty first component)
		String[] components = filename.split(File.separator);
		// If just a simple name with no path, just return the filename
		if ( components.length < 2 )
			return filename;

		// Initialize to "..." + File.separator + final_name
		int numPre = 0;
		int numPost = 1;
		int firstPost = components.length - 1;
		int accumLength = components[firstPost].length() + 4;

		// If this shortest name is no shorter than the full filename, return the full filename
		if ( filename.length() <= accumLength )
			return filename;

		// Construct the truncated name
		while ( (numPre + numPost < components.length) && (accumLength < maxLength) ) {
			if (numPre <= numPost ) {
				accumLength += components[numPre].length() + 1;
				if ( accumLength <= maxLength ) {
					numPre += 1;
				}
			}
			else {
				accumLength += components[firstPost - 1].length() + 1;
				if ( accumLength <= maxLength ) {
					firstPost -= 1;
					numPost += 1;
				}
			}
		}

		// Build the new truncated pathname
		StringBuilder strBuilder = new StringBuilder();
		for (int k = 0; k < numPre; k++) {
			strBuilder.append(components[k]);
			strBuilder.append(File.separator);
		}
		strBuilder.append("...");
		for (int k = firstPost; k < components.length; k++) {
			strBuilder.append(File.separator);
			strBuilder.append(components[k]);
		}

		return strBuilder.toString();
	}
}
