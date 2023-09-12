package com.photonj.imagej;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.photonj.binding.Binding;
import com.photonj.binding.Vector3;
import com.photonj.binding.Binding.Spot;
import com.photonj.binding.IHasLocation;

import ij.IJ;
import ncsa.hdf.hdf5lib.exceptions.HDF5Exception;
import ncsa.hdf.object.Dataset;
import ncsa.hdf.object.Datatype;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.h5.H5Datatype;
import ncsa.hdf.object.h5.H5File;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

public class LocResults {

	int originalWidth,originalHeight;
	public Vector3 scaling; // pixelsizes
	public ArrayList<Spot> results;
	
	public KDTree<Spot> kdTree;
	
	public Vector3[] getPositions() {
		Vector3[] positions = new Vector3[results.size()];
		for (int i=0;i<results.size();i++) {
			positions[i] = results.get(i).getLocation();
		}
		return positions;
	}
	
	public LocResults(String fn)
	{
		try {
			//FileFormat fileFormat = FileFormat.getFileFormat(FileFormat.FILE_TYPE_HDF5);
			H5File file = (H5File) FileFormat.getInstance(fn);
			if (!file.canRead()) {
				IJ.error("Can't read " + fn);
				return;
			}
			file.open();

			/*
			List<Dataset> datasets=getDatasets(file);
			
			for (int i=0;i<datasets.size();i++)
			{
				System.out.println("Dataset: " + datasets.get(i).getFullName());
			}*/
			file.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {}
	}
	
	public void exportPicassoHDF5(String fn)
	{
		/*
         dtype = [('frame', '<u4'), 
         ('x', '<f4'), ('y', '<f4'),
         ('photons', '<f4'), 
         ('sx', '<f4'), ('sy', '<f4'), 
         ('bg', '<f4'), 
         ('lpx', '<f4'), ('lpy', '<f4'), 
         ('lI', '<f4'), ('lbg', '<f4'), 
         ('ellipticity', '<f4'), 
         ('net_gradient', '<f4')]
		 */
		try {
			FileFormat fileFormat = FileFormat.getFileFormat(FileFormat.FILE_TYPE_HDF5);
			H5File file = (H5File) fileFormat.createFile(fn, FileFormat.FILE_CREATE_DELETE);
			if (!file.canWrite()) {
				IJ.error("File `" + fn + "`is readonly!");
				return;
			}
			file.open();
			
			H5Datatype floatType = (H5Datatype) file.createDatatype(Datatype.CLASS_FLOAT,4,Datatype.ORDER_LE,Datatype.NATIVE);
			H5Datatype uintType = (H5Datatype)file.createDatatype(Datatype.CLASS_INTEGER,Datatype.NATIVE,Datatype.ORDER_LE,Datatype.SIGN_NONE);
									
			Datatype[] dtypes= {
					uintType,
					floatType,floatType,floatType,floatType,floatType,floatType,floatType,floatType,floatType,floatType,floatType,floatType
				};
			
			String[] memberNames = {
				"frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "lI", "lbg", "ellipticity", "net_gradient"
			};
//			Group pgroup = 
	//		           (Group)((DefaultMutableTreeNode)getRootNode).getUserObject();
			long[] dims = {results.size()};

			int[] memberSizes = new int[memberNames.length];
			for(int i=0;i<memberNames.length;i++) 
				memberSizes[i]=1;
			
			Dataset dataset=file.createCompoundDS("locs", null, dims, null, null, 0, memberNames, dtypes, memberSizes, null);
			dataset.init();
			
			int n = results.size();
			float[] lx = new float[n],
				ly = new float[n],
				lI = new float[n], 
				lsx = new float[n],
				lsy = new float[n], 
				lbg = new float[n], 
				llpx = new float[n], 
				llpy = new float[n], 
				llI = new float[n], 
				llbg = new float[n], 
				lzero = new float[n];
			int[] frames=new int[n];
						
			int framecount=0;
			for (int i=0;i<results.size();i++) {
				Spot spot=results.get(i);
				lx[i]=spot.x;
				ly[i]=spot.y;
				lI[i]=spot.I;
				lbg[i]=spot.bg;
				llpx[i]=spot.x_crlb;
				llpy[i]=spot.y_crlb;
				llbg[i]=spot.bg_crlb;
				llI[i]=spot.I_crlb;
				frames[i]=spot.frame;
				lsx[i]=spot.sigmaX;
				lsy[i]=spot.sigmaY;
				if(spot.frame>=framecount)
					framecount=spot.frame+1;
			}
			Object[] data=new Object[] { frames,lx,ly,lI,lsx,lsy,lbg,llpx,llpy,llI,llbg,lzero,lzero};
			List<?> datalist= Arrays.asList(data);
			dataset.write(datalist);
			
			file.close();

			String fnWithoutExt = fn.substring(0,fn.lastIndexOf("."));
			
			PrintWriter wr = new PrintWriter(fnWithoutExt + ".yaml");
			
			wr.println("Byte Order: <");
			wr.println("Camera: Dont know");
			wr.println("Data Type: uint16");
			wr.println("File: " + new File(fn).getName());
			wr.println("Frames: " + framecount);
			wr.println("Height: " + originalHeight);
			wr.println("Width: " + originalWidth);
			
			wr.close();
		} catch (HDF5Exception err) {
			IJ.error(err.getMessage());
			return;
		} catch (Exception e) {
			IJ.error(e.getMessage());

			return;
		}

	}
	
	public void exportText(String fn)
	{
		
	}
	
	public void loadHDF5(String fn)
	{
		
	}
	
	
	public LocResults(ArrayList<Spot> results, int orgw, int orgh, Vector3 scaling)
	{
		this.originalHeight=orgh;
		this.originalWidth=orgw;
		this.results=results;
		this.scaling = scaling; // pixelsizes..
		
		ArrayList<IHasLocation> treeData=new ArrayList<>(results);
		
		this.kdTree=new KDTree<Spot>(treeData, 100);
	}
	
		
	public RandomAccessibleInterval<FloatType> renderToImage(int width, int height, Transform transform, float sigma, Binding binding)
    {
		float[] spotX, spotY, spotI, spotSX=null, spotSY=null;
		
		spotX=new float[results.size()];
		spotY=new float[results.size()];
		spotI=new float[results.size()];
		
		if (sigma<0.0f) {
			spotSX=new float[results.size()];
			spotSY=new float[results.size()];
		}
		
		for (int i=0;i<spotX.length;i++) {
			Binding.Spot s = results.get(i);
			// transform all points to the given xy plane
			
			Vector3 pos = transform.transform(new Vector3(s.x,s.y,s.z));
			spotX[i] = pos.x;
			spotY[i] = pos.y;
			spotI[i] = s.I;

			if (spotSX!=null) {
				Vector3 crlb=new Vector3(s.x_crlb,s.y_crlb,s.z_crlb);
				spotSX[i]=crlb.dot(transform.m[0]);// xAxis[0]*s.x_crlb+xAxis[1]*s.y_crlb+xAxis[2]*s.z_crlb;
				spotSY[i]=crlb.dot(transform.m[1]);//yAxis[0]*s.x_crlb+yAxis[1]*s.y_crlb+yAxis[2]*s.z_crlb;
			}
		}
		
		float zoom =  0.5f * ( transform.m[0].len() + transform.m[1].len());
		
		int roisize=(int)(3+4*sigma*zoom);
		if (sigma<0.0f)
			roisize =(int)(3+4*getMeanCRLBX()*zoom);
		
		//System.out.println("rendering roisize: " + roisize);

		float[] renderedImage = binding.RenderGauss2D(width, height, spotX,spotY,spotI,spotSX,spotSY,sigma*zoom, roisize);
		final RandomAccessibleInterval<FloatType> newImage = (RandomAccessibleInterval<FloatType>) ArrayImgs.floats(renderedImage, width, height); 
		
		return newImage;
    }

	public float getMeanCRLBX() {
		float s=0.0f;
		for (Spot spot : results)
			s+=spot.x_crlb;
		return s/results.size();
	}


	// compute [min,max] after applying transformation to each point
	public Vector3[] computeExtent(Transform transform) {
		Vector3[] pos = transform.transform(getPositions());
		if( pos.length > 0)
			return new Vector3[] { Vector3.min(pos), Vector3.max(pos) };
		return null;
	}

	public boolean empty() {
		return results.isEmpty();
	}
}
