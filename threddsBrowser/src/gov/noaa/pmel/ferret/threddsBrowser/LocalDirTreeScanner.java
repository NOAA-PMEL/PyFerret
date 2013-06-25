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

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import javax.swing.SwingWorker;

import thredds.catalog.InvCatalogImpl;
import thredds.catalog.InvDataset;
import thredds.catalog.InvDatasetImpl;
import thredds.catalog.InvService;
import thredds.catalog.ServiceType;

/**
 * Scanner for a local directory to generate a complete catalog of the tree implemented as a SwingWorker.
 * Calling the execute on this SwingWorker will generate a complete catalog of the directory tree rooted 
 * at the local directory given in the constructor that pass the file filter given in the constructor.
 * This catalog can then be retrieved by the get method.  Progress is based on the percentage of the number 
 * of entries in the root directory, possibly plus its immediate subdirectories, that have been examined.
 * Progress is reported (as with all SwingWorkers) by firing PropertyChange events with the property name 
 * "progress" and a new value in [0,100].  The currently directory being examined is reported by firing 
 * PropertyChange events with the property name "Directory" and the File object representing the directory 
 * as the new value.
 * 
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class LocalDirTreeScanner extends SwingWorker<InvCatalogImpl, File> {
	/** root of the local directory tree to scan */
	private File localDir;
	/** filter for the files/directories added to the catalog; can be null */
	private FileFilter datasetFilter;
	/** the progress total number of files/directories in localDir and its subdirectory passing the file filter */
	private int numToExamine;
	/** the subdirectory depth to count entries for progress */
	private int countDepth;
	/** the progress examined number of files/directories in localDir and its subdirectory passing the file filter */
	private int numExamined;
	/** the actual number of files/directories under localDir in the catalog */
	private int actualCount;

	
	/**
	 * Create a scanner/catalog builder ready to scan the local directory tree rooted at localDir
	 * @param localDir local root directory of the tree to scan
	 * @param datasetFilter filter for the files/directories added to the catalog; 
	 * can be null, in which case all files/directories are added
	 * @throws IOException if localDir is not a valid directory
	 */
	public LocalDirTreeScanner(File localDir, FileFilter datasetFilter) throws IOException {
		if ( ! localDir.isDirectory() )
			throw new IOException("Not a valid local directory: " + localDir.getPath());
		this.localDir = localDir;
		this.datasetFilter = datasetFilter;
		numToExamine = 0;
		countDepth = 0;
		numExamined = 0;
		actualCount = 0;
	}

	/**
	 * Generate a complete catalog of the directory tree rooted at the local
	 * directory given in the constructor of this class.
	 */
	@Override
	protected InvCatalogImpl doInBackground() throws Exception {
		// Send notice that we are examining the local tree root directory
		setProgress(0);
		publish(localDir);

		// Initial the progress variables
		computeNumToExamine();
		numExamined = 0;
		if ( isCancelled() )
			return null;

	    // Create the service for the catalog and datasets
	    InvService service = new InvService("file:", ServiceType.FILE.toString(), "file:", null, null);

	    // Create a new catalog
		InvCatalogImpl catalog = new InvCatalogImpl(localDir.getPath(), null, localDir.toURI());
		catalog.addService(service);

		// Create the top-level dataset.  Since an access will be created in finish() 
		// for each dataset with a URI path, don't add URI paths to directories
		InvDatasetImpl topDataset = new LocalDirInvDatasetImpl(null, localDir.getName(), service.getName());
		topDataset.setID(localDir.getPath());
		topDataset.setCatalog(catalog);

		// Recurse into the tree, adding InvDatasetImpl objects to the datasets field
		addContentDatasets(topDataset, localDir, datasetFilter, service, 0);
		if ( isCancelled() )
			return null;
		// Send notice that we are back to the root of the local directory tree
		publish(localDir);

		// Add this dataset to the catalog
		catalog.addDataset(topDataset);

		// Finish construction of the catalog
		if ( ! catalog.finish() )
			throw new IOException("Unable to finish construction of the catalog from " + localDir.getPath());

		// Set the progress value to complete
		setProgress(100);

		return catalog;
	}

	/**
	 * Using the local directory tree root and file filter used in the construction of this object,
	 * assigns numToExamine and countDepth appropriately.  The value of numToExamine will be the 
	 * one (for the root directory) plus the number of files and directories in this root directory, 
	 * possibly plus the number of file and directories in the immediate subdirectories of this root 
	 * directory.  The value of countDepth will be one (if only subdirectories of the root directory
	 * is counted) or two (if subdirectories of these directories are also counted).
	 * @throws IOException if localDir cannot be examined
	 */
	private void computeNumToExamine() throws IOException {
		numToExamine = 1;

		// Get the number of entries in localDir
		File[] dirArray = localDir.listFiles(datasetFilter);
		if ( dirArray == null )
			throw new IOException("Unable to examine " + localDir.getPath());
		numToExamine += dirArray.length;

		// Decide whether to stop at this level or go into its subdirectories
		if ( numToExamine >= 20 ) {
			countDepth = 1;
			return;
		}

		for (File subDir : dirArray) {
			if ( subDir.isDirectory() ) {
				if ( isCancelled() )
					return;

				// Add the number of entries in this subdirectory of localDir
				try {
					File[] subDirArray = subDir.listFiles(datasetFilter);
					if ( subDirArray != null ) {
						numToExamine += subDirArray.length;
					}
				} catch (Exception e) {
					; // don't care
				}
			}
		}
		countDepth = 2;
	}

	/**
	 * Adds datasets to parentDataset for each file/directory in parentDir that passes the filter.
	 * When subdirectories are encountered, this method calls itself with the subdirectory dataset
	 * and subdirectory File, thus filling out the full directory tree rooted at parentDir.
	 * @param parentDataset add new child datasets to this dataset
	 * @param parentDir directory to examine
	 * @param datasetFilter filter on the files/directories to be added to the catalog.  
	 * If null, all files/directories are added.
	 * @param serviceName service name to be added to the InvDatasetImpl objects created
	 * @param level the recursion level
	 * @throws IOException if any of the file system operations throws one or if parentDir 
	 * is unable to be read
	 */
	private void addContentDatasets(InvDatasetImpl parentDataset, File parentDir, FileFilter datasetFilter, 
									InvService service, int level) throws IOException {
		// Get the list of files and directories in this directory
		File[] contentsArray = null;
		try {
			contentsArray = parentDir.listFiles(datasetFilter);
		} catch (SecurityException e) {
			;  // if don't have permission to read the directory, leave contentsArray null
		}

		// If there was a problem getting the contents of this directory, or it is empty, just go continue on to the next item
		if ( (contentsArray == null) || (contentsArray.length == 0) ) {
			return;
		}
		actualCount += contentsArray.length;

		// Sort the array of files and directories
		if ( contentsArray.length > 1 ) {
			List<File> contentsList = Arrays.asList(contentsArray);
			Collections.sort(contentsList);
		}

		// Add to the parent's datasets array a dataset for each file/dir returned
		List<InvDataset> datasets = parentDataset.getDatasets();
		for (File child : contentsArray) {
			if ( isCancelled() )
				return;
			if ( child.isDirectory() ) {
				InvDatasetImpl childDataset;
				// Limited the level of recursion (mainly for infinite loops from symbolic links)
				if ( level < 16 ) {
					// Send notice of the new directory being examine
					publish(child);
					// Create the dataset with a null urlPath argument so no access created
					childDataset = new LocalDirInvDatasetImpl(parentDataset, child.getName(), service.getName());
					childDataset.setID(child.getPath());
					// Add the contents of this directory 
					addContentDatasets(childDataset, child, datasetFilter, service, level + 1);
					// Send notice that we are back to the parent directory
					publish(parentDir);
				}
				else {
					// Create the dataset with a null urlPath argument so no access created
					childDataset = new LocalDirInvDatasetImpl(parentDataset, child.getName() + " (not examined)", service.getName());
					childDataset.setID(child.getPath());					
				}
				// Add this dataset to the parent's dataset
				datasets.add(childDataset);
			}
			else {
				// Get the URI string
				String uriPathString = child.toURI().toString();
				// Remove the base string from the service
				String serviceBaseString = service.getBase();
				if ( ! uriPathString.startsWith(serviceBaseString) )
					throw new IOException("Unexpected URI string of child: " + uriPathString + "\n" +
										  "under service URI string: " + serviceBaseString);
				uriPathString = uriPathString.substring(serviceBaseString.length());
				// Create the dataset with this urlPath argument so an appropriate access is created
				InvDatasetImpl childDataset = new InvDatasetImpl(parentDataset, child.getName(), null, service.getName(), uriPathString);
				childDataset.setID(child.getPath());
				childDataset.setDataSize(child.length());
				childDataset.setLastModifiedDate(new Date(child.lastModified()));
				// Add this dataset to the parent's dataset
				datasets.add(childDataset);
			}
			if ( level < countDepth ) {
				// Update the progress
				numExamined += 1;
				setProgress((100 * numExamined) / numToExamine);
			}
		}
	}

	/**
	 * @return the total number of entries in the returned catalog.  If canceled,
	 * the total number of entries examined prior to cancellation.
	 */
	public int getNumCatalogEntries() {
		return actualCount;
	}

	/**
	 * Receives a lists of the directory Files that have been examined.
	 * A method generates a PropertyChange event with the name "Directory"
	 * and a new value of the last File in the list.
	 */
	@Override
	protected void process(List<File> fileList) {
		firePropertyChange("Directory", null, fileList.get(fileList.size() - 1));
	}
}
