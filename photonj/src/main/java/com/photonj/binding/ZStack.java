package com.photonj.binding;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import org.apache.commons.io.IOUtils;

import com.photonj.binding.Binding.Estimator;
import com.photonj.binding.Binding.ICameraCalibration;

public class ZStack {
	public float[] data;
	public int[] shape;
	public float[] zrange;
	
	public int getSize() {
		int s=1;
		for (int i=0;i<shape.length;i++)
			s*=shape[i];
		return s;
	}
	

	public ZStack(float[] data, int[] shape, float[] zrange) {
		this.data=data;
		this.shape=shape;
		this.zrange=zrange;
	}
	
	public static ZStack readFromFile(String fn) throws IOException {
		FileInputStream stream = new FileInputStream(fn);
		byte[] bytes = IOUtils.toByteArray(stream);
		ByteBuffer bb=ByteBuffer.wrap(bytes);
		bb.order(ByteOrder.LITTLE_ENDIAN); // DataInputStream only supports big endian.... Why?? Why does java suck so much? 
		
		int version = bb.getInt();
		
		if (version != 1) {
			throw new IOException("Invalid file format / bad version");
		}

		int[] shape = new int[3];
		int size=1;
		for(int i=0;i<3;i++) {
			shape[i]=bb.getInt();
			size *= shape[i];
		}

		float[] zrange = new float[] { bb.getFloat(), bb.getFloat() };

		float[] data=new float[size];
		for (int i=0;i<data.length;i++) {
			data[i] = bb.getFloat();
		}
		return new ZStack(data, shape, zrange);
	}


	public static ZStack fromPSF(Estimator estim, int numZSlices) {

		if(estim.numParams() != 5) {
			throw new RuntimeException("ZStack.fromPSF is expected a model with 5 parameters (x,y,z,I,bg)");
		}
		
		// create a list of spot parameters
		float[] params=new float[5 * numZSlices];
		int roisize = estim.roisize();
		
		int ZAxis=2;
		for (int i=0;i<numZSlices;i++) {
			params[i*5 + 0] = roisize * 0.5f;
			params[i*5 + 1] = roisize * 0.5f;
			if (numZSlices == 1)
				params[ZAxis] = estim.minValues[ZAxis]*0.5f + estim.maxValues[ZAxis]*0.5f; // choose middle position if there is only one slice
			else
				params[i*5 + ZAxis] = estim.minValues[ZAxis] + (estim.maxValues[ZAxis]-estim.minValues[ZAxis]) * i/(float)(numZSlices-1);

			params[i*5+3] = 1.0f;
			params[i*5+4] = 0.0f;
		}
		float[] expval = estim.expectedValue(params, numZSlices);		
		
		return new ZStack(expval, 
				new int[] {numZSlices, roisize,roisize}, 
				new float[] {estim.minValues[ZAxis],estim.maxValues[ZAxis]}
		);

	}


	public float sliceToZPos(int z) {
		if (shape[0] == 1)
			return zrange[0];
		return zrange[0] + (zrange[1]-zrange[0]) * (z-1) / (float)shape[0];
	}


	public float[] getSlice(int i) {
		return Arrays.copyOfRange(data, i*shape[1]*shape[2], (i+1)*shape[1]*shape[2]);
	}
}
