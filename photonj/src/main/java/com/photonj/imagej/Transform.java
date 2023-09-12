package com.photonj.imagej;

import com.photonj.binding.Vector3;

public class Transform {
	public Vector3 t;
	public Vector3[] m;
	
	public Transform() {
		m = new Vector3[] {
				new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)
		};
		t  = new Vector3();
	}
	
	public Transform(Vector3[] m, Vector3 t) { 
		this.t=t;
		this.m=m;
	}
	
	public Vector3 transform(Vector3 pos) {
		return Vector3.matrixApply(m, pos).add(t);
	}
	
	public Vector3[] transform(Vector3[] pos) {
		Vector3[] r=new Vector3[pos.length];
		for (int i=0;i<pos.length;i++)
			r[i]=transform(pos[i]);
		return r;
	}
	
	public Transform inverse() {
		Vector3[] invM = Vector3.matrixInverse(m);
		return new Transform(invM, Vector3.matrixApply(invM, t).neg());
	}
	
	public boolean equals(Transform tr) {
		return m[0].equals(tr.m[0]) && 
				m[1].equals(tr.m[1]) && 
				m[2].equals(tr.m[2]) && t.equals(tr.t);
	}
	
	@Override
	public String toString() {
		return String.format("M={%s,%s,%s}; T=%s", m[0], m[1], m[2], t);
	}
}

