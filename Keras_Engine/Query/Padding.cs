/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

using System;
using k = Keras;
using System.Collections.Generic;
using BH.oM.DeepLearning;

namespace BH.Engine.Keras
{
    public static partial class Query
    {
        /***************************************************/
        /**** Public Methods                            ****/
        /***************************************************/

        public static Shape2d Padding(k.Layers.Conv2D conv2d)
        {
            // Note that the division by 2 means that there might be cases
            // when the padding on both sides (top vs bottom, right vs left) are off by one.
            // In this case, the bottom and right sides always get the one additional padded pixel
            Shape3d shape = (conv2d.Parameters["input_shape"] as k.Shape).FromKeras() as Shape3d;

            switch ((conv2d.Parameters["padding"] as string)?.ToLower())
            {
                case "valid":
                    return new Shape2d() { Dim1 = 0, Dim2 = 0 };
                case "same":
                    // ((W - F * 2P)/S) + 1 and ((H - F * 2P)/S) + 1
                    int height = shape.Dim2;
                    int width = shape.Dim3;
                    k.Shape kernelSize = conv2d.Parameters["kernel_size"] as k.Shape;
                    int stride = (int)conv2d.Parameters["strides"];

                    int paddingVertical = Padding(height, kernelSize.Dimensions[0], stride);
                    int paddingHorizontal = Padding(width, kernelSize.Dimensions[1], stride);

                    return new Shape2d() { Dim1 = paddingVertical, Dim2 = paddingHorizontal };
                default:
                    return null;
            }
        }

        /***************************************************/

        public static Shape2d Padding(k.Layers.Conv2DTranspose transposedConv2d)
        {
            // Note that the division by 2 means that there might be cases
            // when the padding on both sides (top vs bottom, right vs left) are off by one.
            // In this case, the bottom and right sides always get the one additional padded pixel
            Shape3d shape = (transposedConv2d.Parameters["input_shape"] as k.Shape).FromKeras() as Shape3d;

            switch ((transposedConv2d.Parameters["padding"] as string)?.ToLower())
            {
                case "valid":
                    return new Shape2d() { Dim1 = 0, Dim2 = 0 };
                case "same":
                    // ((W - F * 2P)/S) + 1 and ((H - F * 2P)/S) + 1
                    int height = shape.Dim2;
                    int width = shape.Dim3;
                    k.Shape kernelSize = transposedConv2d.Parameters["kernel_size"] as k.Shape;
                    int stride = (int)transposedConv2d.Parameters["strides"];

                    int paddingVertical = Padding(height, kernelSize.Dimensions[0], stride);
                    int paddingHorizontal = Padding(width, kernelSize.Dimensions[1], stride);

                    return new Shape2d() { Dim1 = paddingVertical, Dim2 = paddingHorizontal };
                default:
                    return null;
            }
        }

        /***************************************************/

        public static int Padding(int length, int kernelSize, int stride)
        {
            return length - ((kernelSize * 2) / stride) + 1;
        }
        
        /***************************************************/
    }
}

