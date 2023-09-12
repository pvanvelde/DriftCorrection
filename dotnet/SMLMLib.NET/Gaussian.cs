using System;
using System.Runtime.InteropServices;
using System.Security;

namespace SMLMLib
{

	public class PixelCalibration
	{ 
		//todo
	}

	public class GaussianPSF
	{
        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        static extern IntPtr Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, 
            IntPtr scmos, IntPtr ctx);

		public static PSF CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, Context ctx, PixelCalibration pc=null)
		{
			return new PSF(Gauss2D_CreatePSF_XYIBg(roisize, sigmaX, sigmaY, cuda, IntPtr.Zero, ctx.Instance));
		}

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda,
            IntPtr scmos, IntPtr ctx);


        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct Gauss3D_Calibration
        {
            fixed float x[4];
            fixed float y[4];
            float minz, maxz;
        };


        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Gauss2D_CreatePSF_XYZIBg(int roisize, 
            ref Gauss3D_Calibration calib, bool cuda, IntPtr scmos, IntPtr ctx);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda,
            IntPtr scmos, IntPtr ctx);

        // Fit X,Y,intensity, and with a fixed background supplied as constant
        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Gauss2D_CreatePSF_XYI(int roisize, float sigma, bool cuda,
            IntPtr scmos, IntPtr ctx);

    }
}
