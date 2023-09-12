using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SMLMLib
{
	[StructLayout(LayoutKind.Sequential)]
	public struct Vector4
	{
		public float x, y, z, w;
		public Vector4(float X = 0.0f, float Y = 0.0f, float Z=0.0f, float W=0.0f)
		{
			x = X; y = Y;
			z = Z;w = W;
		}
	}
}
